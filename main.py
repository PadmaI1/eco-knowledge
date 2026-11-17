
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
