"""
Ingestion pipeline test — extraction → chunking → tokenization → embedding
Prints a detailed stats report for all 3 PDFs.
"""

import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Imports ──────────────────────────────────────────────────────────────────
from src.ingestion.pdf_extractor import extract_all_pdfs
from src.ingestion.chunker import chunk_documents
from src.ingestion.tokenizer import tokenize_chunks
from src.ingestion.embedder import embed_chunks

# ── Config ───────────────────────────────────────────────────────────────────
RAW_DIR        = "data/raw"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 64))

# ── Separator helper ─────────────────────────────────────────────────────────
SEP = "=" * 65

def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ─────────────────────────────────────────────────────────────────────────────

def main():
    section("STEP 1 — PDF EXTRACTION")
    docs = extract_all_pdfs(RAW_DIR)
    for doc in docs:
        total_chars = sum(p.char_count for p in doc.pages)
        non_empty   = sum(1 for p in doc.pages if p.text.strip())
        methods     = defaultdict(int)
        for p in doc.pages:
            methods[p.extraction_method] += 1
        print(
            f"  {doc.source_file}\n"
            f"    pages        : {doc.total_pages} total, {non_empty} non-empty\n"
            f"    chars        : {total_chars:,}\n"
            f"    extraction   : {dict(methods)}"
        )

    # ── Chunking ─────────────────────────────────────────────────────────────
    section("STEP 2 — SEMANTIC CHUNKING (LlamaIndex)")
    all_chunks = chunk_documents(
        docs,
        openai_api_key=OPENAI_API_KEY,
        embedding_model=EMBED_MODEL,
    )

    chunks_by_file = defaultdict(list)
    for c in all_chunks:
        chunks_by_file[c.source_file].append(c)

    for fname, chunks in chunks_by_file.items():
        char_counts = [c.char_count for c in chunks]
        print(
            f"  {fname}\n"
            f"    chunks       : {len(chunks)}\n"
            f"    chars/chunk  : min={min(char_counts)}  "
            f"max={max(char_counts)}  avg={int(sum(char_counts)/len(char_counts))}"
        )
    print(f"\n  TOTAL chunks   : {len(all_chunks)}")

    # ── Tokenization ─────────────────────────────────────────────────────────
    section("STEP 3 — TOKENIZATION (tiktoken cl100k_base)")
    all_chunks = tokenize_chunks(all_chunks)

    for fname, chunks in chunks_by_file.items():
        token_counts = [c.metadata["token_count"] for c in chunks]
        truncated    = sum(1 for c in chunks if c.metadata.get("truncated"))
        print(
            f"  {fname}\n"
            f"    tokens/chunk : min={min(token_counts)}  "
            f"max={max(token_counts)}  avg={int(sum(token_counts)/len(token_counts))}\n"
            f"    total tokens : {sum(token_counts):,}\n"
            f"    truncated    : {truncated}"
        )

    all_tokens = sum(c.metadata["token_count"] for c in all_chunks)
    print(f"\n  TOTAL tokens   : {all_tokens:,}")
    est_cost = (all_tokens / 1_000_000) * 0.02   # $0.02 per 1M tokens
    print(f"  Est. embed cost: ${est_cost:.4f}  (text-embedding-3-small @ $0.02/1M)")

    # ── Embedding ────────────────────────────────────────────────────────────
    section("STEP 4 — EMBEDDING (OpenAI text-embedding-3-small)")
    print("  Calling OpenAI API — this may take a moment ...")
    all_chunks = embed_chunks(
        all_chunks,
        openai_api_key=OPENAI_API_KEY,
        model=EMBED_MODEL,
    )

    embedded = sum(1 for c in all_chunks if "embedding" in c.metadata)
    dim      = len(all_chunks[0].metadata["embedding"]) if embedded else 0
    print(
        f"\n  Embedded       : {embedded}/{len(all_chunks)} chunks\n"
        f"  Vector dim     : {dim}"
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    section("SUMMARY")
    for fname, chunks in chunks_by_file.items():
        token_counts = [c.metadata["token_count"] for c in chunks]
        print(
            f"  {fname}\n"
            f"    chunks  : {len(chunks)}\n"
            f"    tokens  : {sum(token_counts):,}\n"
            f"    avg tok : {int(sum(token_counts)/len(token_counts))}"
        )
    print(f"\n  {'─'*40}")
    print(f"  TOTAL  chunks  : {len(all_chunks)}")
    print(f"  TOTAL  tokens  : {all_tokens:,}")
    print(f"  TOTAL  embedded: {embedded}")
    print(f"  Vector dim     : {dim}")
    print(f"  Est. API cost  : ${est_cost:.4f}")
    print(f"\n{SEP}\n  Pipeline test PASSED\n{SEP}\n")


if __name__ == "__main__":
    main()
