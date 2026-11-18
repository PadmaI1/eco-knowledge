
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


