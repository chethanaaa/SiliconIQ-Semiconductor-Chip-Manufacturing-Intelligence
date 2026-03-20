"""
Hybrid Retrieval
----------------
Combines dense (FAISS cosine) and sparse (BM25) retrieval via
Reciprocal Rank Fusion (RRF), then passes fused candidates to the reranker.

Pipeline per query:
    1. Dense  : embed query → FAISS top-K  (semantic similarity)
    2. Sparse : BM25 top-K                 (exact / lexical match — great for
                                            entity names, process nodes, acronyms)
    3. Fusion : RRF merges both ranked lists into a single scored list
    4. Rerank : hybrid / LLM reranker refines the fused top-K

RRF formula:  rrf_score(d) = Σ  1 / (k + rank(d))
              k=60 is the standard constant that dampens high-rank outliers.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from src.ingestion.embedder import embed_query
from src.rag.reranker import rerank_results


# ── Constants ─────────────────────────────────────────────────────────────────

_RRF_K = 60
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "what", "which", "with", "within", "was", "has", "have",
}
_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9\-_.]+\b")


# ── Tokenizer (shared by BM25 index + query) ─────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase with stopword removal — preserves domain terms like EUV, TSMC, 3nm."""
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS
    ]


# ── BM25 index build / persist / load ────────────────────────────────────────

def build_bm25_index(metadata: List[Dict[str, Any]]) -> BM25Okapi:
    """Build a BM25Okapi index from chunk texts stored in the metadata list."""
    corpus = [_tokenize(doc.get("text", "")) for doc in metadata]
    index = BM25Okapi(corpus)
    logger.info(f"BM25 index built — {len(corpus)} documents")
    return index


def save_bm25_index(index: BM25Okapi, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)
    logger.success(f"BM25 index saved → {path}")


def load_bm25_index(path: str) -> BM25Okapi:
    if not Path(path).exists():
        raise FileNotFoundError(f"BM25 index not found at {path}")
    with open(path, "rb") as f:
        index = pickle.load(f)
    logger.info(f"BM25 index loaded from {path}")
    return index


# ── Dense retrieval (FAISS) ───────────────────────────────────────────────────

def dense_search(
    query_vector: List[float],
    faiss_index: faiss.Index,
    top_k: int,
) -> List[Tuple[int, float]]:
    """
    Search FAISS index with a pre-normalised query vector.
    Returns list of (metadata_index, cosine_score) sorted descending.
    """
    query = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query)
    scores, indices = faiss_index.search(query, top_k)

    return [
        (int(idx), float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]


# ── Sparse retrieval (BM25) ───────────────────────────────────────────────────

def sparse_search(
    query: str,
    bm25_index: BM25Okapi,
    top_k: int,
) -> List[Tuple[int, float]]:
    """
    BM25 search over tokenised chunk texts.
    Returns list of (metadata_index, bm25_score) sorted descending.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = bm25_index.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (int(idx), float(scores[idx]))
        for idx in top_indices
        if scores[idx] > 0.0
    ]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _retrieval_tag(
    idx: int,
    dense_map: Dict[int, int],
    sparse_map: Dict[int, int],
) -> str:
    if idx in dense_map and idx in sparse_map:
        return "both"
    return "dense_only" if idx in dense_map else "sparse_only"


def reciprocal_rank_fusion(
    dense_results: List[Tuple[int, float]],
    sparse_results: List[Tuple[int, float]],
    metadata: List[Dict[str, Any]],
    top_k: int,
    rrf_k: int = _RRF_K,
) -> List[Dict[str, Any]]:
    """
    Merge dense + sparse ranked lists using RRF.
    rrf_score(d) = 1/(rrf_k + rank_dense) + 1/(rrf_k + rank_sparse)

    Returns top_k metadata dicts enriched with:
        dense_score, dense_rank, sparse_score, sparse_rank,
        rrf_score, retrieval_method
    """
    rrf_scores: Dict[int, float] = {}
    dense_rank_map: Dict[int, int] = {}
    sparse_rank_map: Dict[int, int] = {}
    dense_score_map: Dict[int, float] = {}
    sparse_score_map: Dict[int, float] = {}

    for rank, (idx, score) in enumerate(dense_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        dense_rank_map[idx] = rank
        dense_score_map[idx] = score

    for rank, (idx, score) in enumerate(sparse_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        sparse_rank_map[idx] = rank
        sparse_score_map[idx] = score

    sorted_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

    fused: List[Dict[str, Any]] = []
    for idx in sorted_indices[:top_k]:
        doc = dict(metadata[idx])
        doc["rrf_score"]        = round(rrf_scores[idx], 6)
        doc["dense_score"]      = round(dense_score_map.get(idx, 0.0), 6)
        doc["dense_rank"]       = dense_rank_map.get(idx)
        doc["sparse_score"]     = round(sparse_score_map.get(idx, 0.0), 6)
        doc["sparse_rank"]      = sparse_rank_map.get(idx)
        doc["score"]            = doc["rrf_score"]   # unified key for reranker
        doc["retrieval_method"] = _retrieval_tag(idx, dense_rank_map, sparse_rank_map)
        fused.append(doc)

    logger.debug(
        f"RRF fusion — dense:{len(dense_results)} + sparse:{len(sparse_results)} "
        f"→ {len(fused)} unique candidates"
    )
    return fused


# ── Full hybrid retrieve → rerank pipeline ────────────────────────────────────

def retrieve_and_rerank(
    query: str,
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    metadata: List[Dict[str, Any]],
    openai_api_key: str,
    embedding_model: str = "text-embedding-3-small",
    top_k_retrieval: int = 20,
    top_k_rerank: int = 5,
    rerank_strategy: str = "hybrid",
    rerank_api_key: Optional[str] = None,
    rerank_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    End-to-end hybrid retrieval + reranking for a single query.

    Args:
        query            : raw user query string
        faiss_index      : loaded FAISS index
        bm25_index       : loaded BM25 index
        metadata         : parallel metadata list from vector_store
        openai_api_key   : used to embed the query
        embedding_model  : must match the model used at index build time
        top_k_retrieval  : candidates fetched per retriever before fusion
        top_k_rerank     : final results returned after reranking
        rerank_strategy  : "hybrid" | "openai_llm" | "anthropic_llm"
        rerank_api_key   : required for LLM reranking strategies
        rerank_model     : LLM model name for LLM reranking

    Returns:
        Reranked list of chunk dicts, each containing:
            text, source_file, page_num, topic, domain_tags,
            rrf_score, dense_score, sparse_score, retrieval_method,
            rerank_score, rerank_strategy
    """
    logger.info(
        f"Hybrid retrieval | query='{query[:80]}' | "
        f"top_k_retrieval={top_k_retrieval} | top_k_rerank={top_k_rerank}"
    )

    # 1. Embed query (single API call)
    query_vector = embed_query(
        query=query,
        openai_api_key=openai_api_key,
        model=embedding_model,
    )

    # 2. Dense retrieval
    dense_results = dense_search(query_vector, faiss_index, top_k_retrieval)
    logger.debug(f"Dense  retrieved: {len(dense_results)} candidates")

    # 3. Sparse retrieval
    sparse_results = sparse_search(query, bm25_index, top_k_retrieval)
    logger.debug(f"Sparse retrieved: {len(sparse_results)} candidates")

    # 4. RRF fusion
    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        metadata=metadata,
        top_k=top_k_retrieval * 2,   # over-fetch before rerank trims to top_k_rerank
    )

    # 5. Rerank
    reranked = rerank_results(
        query=query,
        candidates=fused,
        top_k=top_k_rerank,
        strategy=rerank_strategy,
        provider_api_key=rerank_api_key,
        model=rerank_model,
    )

    # Log retrieval method mix in final results
    methods = [r.get("retrieval_method", "unknown") for r in reranked]
    method_counts = {m: methods.count(m) for m in set(methods)}
    logger.success(
        f"Final {len(reranked)} results | "
        f"retrieval mix={method_counts} | rerank={rerank_strategy}"
    )

    return reranked
