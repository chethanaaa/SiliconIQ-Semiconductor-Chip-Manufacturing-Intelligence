"""
Vector Store (FAISS)
--------------------
Builds, persists, and loads a FAISS index from embedded TextChunks.

Index type: IndexFlatIP (inner product on L2-normalised vectors = cosine similarity)

Layout on disk:
    data/vector_store/
        faiss_index.bin      — FAISS binary index
        metadata.json        — parallel list of chunk metadata (no embeddings)
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from loguru import logger

from src.ingestion.chunker import TextChunk


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_matrix(chunks: List[TextChunk]) -> np.ndarray:
    """Stack embeddings into a float32 matrix (N, dim)."""
    vectors = [c.metadata["embedding"] for c in chunks]
    matrix = np.array(vectors, dtype=np.float32)
    # L2-normalise so IndexFlatIP computes cosine similarity
    faiss.normalize_L2(matrix)
    return matrix


def _strip_embeddings(chunk: TextChunk) -> dict:
    """Return chunk metadata without the embedding vector (saves disk space)."""
    meta = dict(chunk.metadata)
    meta.pop("embedding", None)
    meta["text"] = chunk.text          # keep text alongside metadata
    return meta


# ── Build ────────────────────────────────────────────────────────────────────

def build_index(chunks: List[TextChunk]) -> Tuple[faiss.Index, List[dict]]:
    """
    Build a FAISS IndexFlatIP from embedded chunks.

    Returns:
        index    : FAISS index ready for similarity search
        metadata : parallel list of dicts (one per chunk, no embedding vectors)
    """
    if not chunks:
        raise ValueError("No chunks provided to build_index")

    # Validate all chunks are embedded
    missing = [c.chunk_id for c in chunks if "embedding" not in c.metadata]
    if missing:
        raise ValueError(f"{len(missing)} chunks are missing embeddings: {missing[:3]} ...")

    matrix = _extract_matrix(chunks)
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    metadata = [_strip_embeddings(c) for c in chunks]

    logger.success(
        f"FAISS index built — {index.ntotal} vectors | dim={dim} | "
        f"index_type=IndexFlatIP (cosine)"
    )
    return index, metadata


# ── Persist ──────────────────────────────────────────────────────────────────

def save_index(
    index: faiss.Index,
    metadata: List[dict],
    index_path: str,
    metadata_path: str,
) -> None:
    """Persist the FAISS index and metadata to disk."""
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    index_mb = os.path.getsize(index_path) / 1024 / 1024
    logger.success(
        f"Saved FAISS index → {index_path} ({index_mb:.1f} MB) | "
        f"metadata → {metadata_path}"
    )


# ── Load ─────────────────────────────────────────────────────────────────────

def load_index(
    index_path: str,
    metadata_path: str,
) -> Tuple[faiss.Index, List[dict]]:
    """Load a previously saved FAISS index and its metadata from disk."""
    if not Path(index_path).exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(
        f"Loaded FAISS index — {index.ntotal} vectors from {index_path}"
    )
    return index, metadata


# ── Search ───────────────────────────────────────────────────────────────────

def search_index(
    index: faiss.Index,
    metadata: List[dict],
    query_vector: List[float],
    top_k: int = 10,
) -> List[dict]:
    """
    Search the FAISS index with a query embedding.

    Args:
        index        : loaded FAISS index
        metadata     : parallel metadata list
        query_vector : raw embedding from OpenAI (will be L2-normalised)
        top_k        : number of results to return

    Returns:
        List of metadata dicts with an added "score" (cosine similarity) field,
        sorted descending by score.
    """
    query = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query)

    scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:           # FAISS returns -1 for unfilled slots
            continue
        result = dict(metadata[idx])
        result["score"] = float(score)
        results.append(result)

    return results
