
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

