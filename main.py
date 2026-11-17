
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

