
from pathlib import Path
import os
import re

from dotenv import load_dotenv
from google import genai

import chromadb
from rank_bm25 import BM25Okapi

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings

from google.genai import types

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# =========================================================
# 1. INITIAL SETUP
# =========================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Separate Gemini client for Ragas evaluation
ragas_client = genai.Client(
    http_options=types.HttpOptions(
        api_version="v1beta"
    )
)

ragas_llm = llm_factory(
    "gemini-2.5-flash",
    provider="google",
    client=ragas_client
)

ragas_embeddings = GoogleEmbeddings(
    client=ragas_client,
    model="gemini-embedding-001"
)

faithfulness_metric = Faithfulness(llm=ragas_llm)

answer_relevancy_metric = AnswerRelevancy(
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

context_precision_metric = ContextPrecision(llm=ragas_llm)

context_recall_metric = ContextRecall(llm=ragas_llm)

chroma_client = chromadb.PersistentClient(
    path="data/chroma"
)

# Set this to True ONCE if you want to rebuild the index
# from scratch. Then change it back to False.
RESET_INDEX = False

if RESET_INDEX:
    try:
        chroma_client.delete_collection("eco_knowledge")
    except Exception:
        pass

collection = chroma_client.get_or_create_collection(
    name="eco_knowledge",
    metadata={"hnsw:space": "cosine"}
)


# =========================================================
# 2. DOCUMENT PROCESSING
# =========================================================

def load_docs(directory):

    documents = []

    txt_files = directory.glob("*.txt")

    for txt_file in txt_files:

        text = txt_file.read_text(encoding="utf-8")

        cleaned_text = " ".join(text.split())

        documents.append({
            "text": cleaned_text,
            "metadata": {
                "source": txt_file.name
            }
        })

    return documents


def get_topic(filename):

    topics = {
        "solar_energy.txt": "renewable_energy",
        "climate_change.txt": "climate",
        "waste_management.txt": "waste"
    }

    return topics.get(filename, "other")


def chunk_documents(documents, chunk_size=200, overlap=40):

    chunks = []

    for document in documents:

        start = 0
        chunk_id = 0

        text = document["text"]

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": document["metadata"]["source"],
                    "chunk_id": chunk_id,
                    "topic": get_topic(
                        document["metadata"]["source"]
                    )
                }
            })

            chunk_id += 1
            start += chunk_size - overlap

    return chunks


# =========================================================
# 3. GENERATE AND STORE EMBEDDINGS
# =========================================================

def generate_embeddings(chunks, client):

    for chunk in chunks:

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk["text"]
        )

        chunk["embedding"] = result.embeddings[0].values

    return chunks


def store_chunks(chunks, collection):

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in chunks:

        source = chunk["metadata"]["source"]
        chunk_id = chunk["metadata"]["chunk_id"]

        ids.append(f"{source}_{chunk_id}")
        documents.append(chunk["text"])
        embeddings.append(chunk["embedding"])
        metadatas.append(chunk["metadata"])

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


# =========================================================
# 4. BUILD THE KNOWLEDGE BASE
# =========================================================

def build_knowledge_base(collection, client):

    # Don't regenerate embeddings if data is already stored.
    if collection.count() > 0:

        print("Knowledge base already exists.")
        print("Total records:", collection.count())

        return

    print("Building knowledge base...")

    documents = load_docs(Path("data/raw"))

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        overlap=40
    )

    embedded_chunks = generate_embeddings(
        chunks,
        client
    )

    store_chunks(
        embedded_chunks,
        collection
    )

    print("Knowledge base created.")
    print("Total records:", collection.count())


# =========================================================
# 5. BM25 KEYWORD SEARCH
# =========================================================

def tokenize(text):

    return re.findall(
        r"[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)*",
        text.lower()
    )


def build_bm25_index(collection):

    data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = data["documents"]
    ids = data["ids"]
    metadatas = data["metadatas"]

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    return bm25, documents, ids, metadatas


def keyword_search(
    question,
    bm25,
    documents,
    ids,
    metadatas,
    top_k=5,
    topic=None
):

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for i in ranked_indices:

        # Ignore documents that don't match the topic.
        if topic and metadatas[i].get("topic") != topic:
            continue

        if scores[i] <= 0:
            continue

        results.append({
            "id": ids[i],
            "text": documents[i],
            "metadata": metadatas[i],
            "score": float(scores[i])
        })

        if len(results) == top_k:
            break

    return results


# =========================================================
# 6. SEMANTIC + KEYWORD SEARCH
# =========================================================

def generate_query_embedding(question, client):

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    return result.embeddings[0].values


def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k=60
):

    scores = {}
    result_lookup = {}

    # Add scores from semantic search.
    for rank, result in enumerate(semantic_results, start=1):

        document_id = result["id"]

        scores[document_id] = (
            scores.get(document_id, 0) + 1 / (k + rank)
        )

        result_lookup[document_id] = result

    # Add scores from keyword search.
    for rank, result in enumerate(keyword_results, start=1):

        document_id = result["id"]

        scores[document_id] = (
            scores.get(document_id, 0) + 1 / (k + rank)
        )

        result_lookup[document_id] = result

    # Sort documents by their combined RRF score.
    ranked_ids = sorted(
        scores,
        key=lambda document_id: scores[document_id],
        reverse=True
    )

    fused_results = []

    for document_id in ranked_ids:

        result = result_lookup[document_id].copy()

        result["rrf_score"] = scores[document_id]

        fused_results.append(result)

    return fused_results


