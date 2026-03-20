"""
Ingestion Pipeline Runner
-------------------------
Runs the full RAG ingestion pipeline and persists indexes to disk.

Steps:
  1. Extract text from all PDFs in data/raw/
  2. Semantic chunking (LlamaIndex)
  3. Tokenize (tiktoken)
  4. Enrich metadata
  5. Embed (OpenAI text-embedding-3-small)
  6. Build + save FAISS index  → data/vector_store/faiss_index.bin
  7. Build + save BM25 index   → data/vector_store/bm25_index.pkl
  8. Save metadata             → data/vector_store/metadata.json

Run this once before starting the app:
    python run_ingestion.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from src.ingestion.pdf_extractor import extract_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.ingestion.tokenizer import tokenize_chunks
from src.ingestion.metadata_enricher import enrich_chunks
from src.ingestion.embedder import embed_chunks
from src.rag.vector_store import build_index, save_index
from src.rag.retrieval import build_bm25_index, save_bm25_index

# ── Config ────────────────────────────────────────────────────────────────────

RAW_DIR        = os.getenv("PDF_RAW_DIR", "data/raw")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/vector_store/faiss_index.bin")
METADATA_PATH    = os.getenv("FAISS_METADATA_PATH", "data/vector_store/metadata.json")
BM25_INDEX_PATH  = "data/vector_store/bm25_index.pkl"

SEP = "=" * 65

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    section("STEP 1 — PDF EXTRACTION")
    docs = extract_all_pdfs(RAW_DIR)
    total_pages = sum(d.total_pages for d in docs)
    print(f"  {len(docs)} PDFs | {total_pages} pages total")

    section("STEP 2 — SEMANTIC CHUNKING")
    chunks = chunk_documents(docs, openai_api_key=OPENAI_API_KEY, embedding_model=EMBED_MODEL)
    print(f"  {len(chunks)} semantic chunks")

    section("STEP 3 — TOKENIZATION")
    chunks = tokenize_chunks(chunks)

    section("STEP 4 — METADATA ENRICHMENT")
    chunks = enrich_chunks(chunks)

    section("STEP 5 — EMBEDDING")
    chunks = embed_chunks(chunks, openai_api_key=OPENAI_API_KEY, model=EMBED_MODEL)

    section("STEP 6 — BUILD & SAVE FAISS INDEX")
    faiss_index, metadata = build_index(chunks)
    save_index(faiss_index, metadata, FAISS_INDEX_PATH, METADATA_PATH)

    section("STEP 7 — BUILD & SAVE BM25 INDEX")
    bm25_index = build_bm25_index(metadata)
    save_bm25_index(bm25_index, BM25_INDEX_PATH)

    section("INGESTION COMPLETE")
    print(f"  Chunks indexed : {len(chunks)}")
    print(f"  FAISS index    : {FAISS_INDEX_PATH}")
    print(f"  BM25 index     : {BM25_INDEX_PATH}")
    print(f"  Metadata       : {METADATA_PATH}")
    print(f"\n  Run the app: streamlit run app/main.py\n")


if __name__ == "__main__":
    main()
