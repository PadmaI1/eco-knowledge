<p align="center">
  <img src="assets/logo.svg" alt="EcoKnowledge logo" width="140">
</p>

<h1 align="center">EcoKnowledge</h1>

<p align="center">
  A retrieval-augmented question-answering system for environmental topics, built with Gemini, ChromaDB, hybrid search and RAGAS evaluation.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white">
  <img alt="Gemini" src="https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4?logo=google&logoColor=white">
  <img alt="ChromaDB" src="https://img.shields.io/badge/vector%20store-ChromaDB-2E7D4F">
  <img alt="uv" src="https://img.shields.io/badge/packaged%20with-uv-DE5FE9">
</p>

---

## Overview

EcoKnowledge answers questions about climate change, solar energy and waste management using a small curated knowledge base. Every answer is grounded in retrieved document chunks and comes with source citations. The pipeline combines semantic and keyword retrieval, reranks the results with a cross-encoder, and is evaluated with both retrieval metrics and RAGAS.

## Features

- **Document ingestion** – loads text files, tags them by topic and splits them into overlapping chunks (200 characters, 40-word overlap).
- **Embeddings and vector store** – embeds chunks with `gemini-embedding-001` and stores them in a persistent ChromaDB collection (cosine distance).
- **Hybrid search** – combines ChromaDB semantic search with BM25 keyword search using Reciprocal Rank Fusion.
- **Cross-encoder reranking** – reranks fused results with `cross-encoder/ms-marco-MiniLM-L-6-v2` and keeps the top 3.
- **Grounded answers with citations** – generates answers with `gemini-2.5-flash` from the retrieved context only, and returns the source files used.
- **Prompt-injection safeguards** – the prompt tells the model to treat documents as data, never follow instructions inside them and never reveal system instructions. A dedicated test document checks this.
- **Evaluation**
  - Retrieval precision and recall per question.
  - RAGAS metrics: faithfulness, answer relevancy, context precision and context recall.

## How it works

```
Question
   │
   ├──► Semantic search (Gemini embeddings + ChromaDB) ─┐
   │                                                    ├──► Reciprocal Rank Fusion ──► Cross-encoder rerank (top 3)
   └──► Keyword search (BM25) ──────────────────────────┘                                      │
                                                                                               ▼
                                                    Answer + sources ◄── Gemini 2.5 Flash ◄── Grounded prompt
```

## Project structure

```
EcoKnowledge/
├── assets/logo.svg          # Project logo
├── data/
│   ├── raw/                 # Knowledge base documents (.txt)
│   ├── chroma/              # Persistent ChromaDB store (generated, git-ignored)
│   └── processed/           # Generated embeddings (git-ignored)
├── src/ecoknowledge/        # Package skeleton
├── main.py                  # RAG pipeline and evaluation
├── eval.py                  # Standalone precision/recall example
├── pyproject.toml
└── uv.lock
```

## Getting started

### Prerequisites

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- A Google Gemini API key

### Installation

```bash
git clone https://github.com/<your-username>/EcoKnowledge.git
cd EcoKnowledge
uv sync
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

### Run

```bash
uv run main.py
```

On the first run the knowledge base is built from `data/raw/` and stored in `data/chroma/`. Later runs reuse the stored index. To rebuild it from scratch, set `RESET_INDEX = True` in `main.py` for one run and then set it back to `False`.

The script builds the BM25 index, then runs the retrieval evaluation and the RAGAS evaluation over the built-in evaluation dataset.

### Adding documents

Drop new `.txt` files into `data/raw/`, then rebuild the index with `RESET_INDEX = True`.

## Evaluation

| Metric | What it measures |
|---|---|
| Precision / Recall | Whether the retrieved sources match the expected sources |
| Faithfulness | Whether the answer is supported by the retrieved context |
| Answer relevancy | Whether the answer addresses the question |
| Context precision | Whether the relevant chunks are ranked highly |
| Context recall | Whether the context covers the reference answer |

## Tech stack

- [Google Gemini](https://ai.google.dev/) (`google-genai`) for generation and embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) for keyword search
- [sentence-transformers](https://www.sbert.net/) for cross-encoder reranking
- [RAGAS](https://docs.ragas.io/) for evaluation
- [uv](https://docs.astral.sh/uv/) for dependency management
