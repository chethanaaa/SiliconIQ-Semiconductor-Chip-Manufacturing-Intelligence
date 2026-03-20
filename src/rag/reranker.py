"""
Reranker
--------
Reorders FAISS retrieval candidates before they are passed to generation.

Strategies:
- hybrid          : local score combining vector similarity + lexical overlap
                    + semiconductor metadata alignment
- openai_llm      : optional listwise reranking with an OpenAI chat model
- anthropic_llm   : optional listwise reranking with an Anthropic model

The hybrid strategy is deterministic, fast, and works without extra network
dependencies beyond embeddings/vector search. LLM reranking is available as a
second stage when higher precision is needed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from loguru import logger
from openai import OpenAI


_WORD_RE = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9\-_]+\b")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "what",
    "which", "with", "within",
}


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in _WORD_RE.findall(text)
        if token.lower() not in _STOPWORDS
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _lexical_overlap_score(query: str, document_text: str) -> float:
    query_terms = set(_tokenize(query))
    doc_terms = set(_tokenize(document_text))
    if not query_terms or not doc_terms:
        return 0.0

    overlap = query_terms & doc_terms
    if not overlap:
        return 0.0

    precision = len(overlap) / len(doc_terms)
    recall = len(overlap) / len(query_terms)
    return (2 * precision * recall) / max(precision + recall, 1e-9)


def _metadata_alignment_score(query: str, candidate: Dict[str, Any]) -> float:
    query_lower = query.lower()
    score = 0.0

    section_header = str(candidate.get("section_header") or "").lower()
    if section_header and section_header in query_lower:
        score += 0.35

    topic = str(candidate.get("topic") or "").replace("_", " ").lower()
    if topic and topic != "general":
        topic_terms = set(topic.split())
        if topic_terms and topic_terms.intersection(_tokenize(query)):
            score += 0.30

    doc_type = str(candidate.get("doc_type") or "").replace("_", " ").lower()
    if doc_type and doc_type in query_lower:
        score += 0.20

    domain_tags = [str(tag).lower() for tag in candidate.get("domain_tags", [])]
    if domain_tags:
        tag_hits = sum(1 for tag in domain_tags if tag in query_lower)
        score += min(tag_hits * 0.12, 0.36)

    return min(score, 1.0)


def _hybrid_score(query: str, candidate: Dict[str, Any]) -> float:
    vector_score = max(_safe_float(candidate.get("score")), 0.0)
    lexical_score = _lexical_overlap_score(query, str(candidate.get("text") or ""))
    metadata_score = _metadata_alignment_score(query, candidate)

    # FAISS cosine score is primary; lexical and metadata make retrieval sharper
    # for semiconductor-domain questions with specific entities/process steps.
    return (
        (0.55 * vector_score)
        + (0.30 * lexical_score)
        + (0.15 * metadata_score)
    )


def _build_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    lines = [
        "You are reranking retrieval candidates for a semiconductor supply-chain RAG system.",
        "Return JSON only in the form:",
        '{"ranked_ids": ["id1", "id2"], "reasoning": {"id1": "short reason"}}',
        "Rank the most relevant chunks first for answering the query.",
        f"Query: {query}",
        "Candidates:",
    ]

    for idx, candidate in enumerate(candidates, start=1):
        chunk_id = candidate.get("chunk_id", f"candidate_{idx}")
        preview = " ".join(str(candidate.get("text", "")).split())[:900]
        lines.append(
            (
                f"[{chunk_id}] source={candidate.get('source_file')} "
                f"page={candidate.get('page_num')} "
                f"topic={candidate.get('topic')} "
                f"doc_type={candidate.get('doc_type')} "
                f"vector_score={_safe_float(candidate.get('score')):.4f}\n"
                f"{preview}"
            )
        )
    return "\n\n".join(lines)


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in reranker response")
    return json.loads(match.group(0))


def _rerank_with_openai(
    query: str,
    candidates: List[Dict[str, Any]],
    api_key: str,
    model: str,
) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(query, candidates)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    payload = _extract_json(response.choices[0].message.content)
    return _apply_llm_ranking(candidates, payload)


def _rerank_with_anthropic(
    query: str,
    candidates: List[Dict[str, Any]],
    api_key: str,
    model: str,
) -> List[Dict[str, Any]]:
    client = Anthropic(api_key=api_key)
    prompt = _build_prompt(query, candidates)
    response = client.messages.create(
        model=model,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    payload = _extract_json(text)
    return _apply_llm_ranking(candidates, payload)


def _apply_llm_ranking(
    candidates: List[Dict[str, Any]],
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ranked_ids = payload.get("ranked_ids", [])
    reasoning = payload.get("reasoning", {})
    candidate_by_id = {
        candidate.get("chunk_id", f"candidate_{idx}"): candidate
        for idx, candidate in enumerate(candidates, start=1)
    }

    reranked: List[Dict[str, Any]] = []
    seen = set()
    total = max(len(ranked_ids), 1)

    for position, chunk_id in enumerate(ranked_ids):
        candidate = candidate_by_id.get(chunk_id)
        if not candidate:
            continue
        item = dict(candidate)
        item["rerank_score"] = round(1 - (position / total), 6)
        item["rerank_reason"] = reasoning.get(chunk_id)
        item["rerank_strategy"] = "llm"
        reranked.append(item)
        seen.add(chunk_id)

    for candidate in candidates:
        chunk_id = candidate.get("chunk_id")
        if chunk_id in seen:
            continue
        item = dict(candidate)
        item["rerank_score"] = 0.0
        item["rerank_strategy"] = "llm_fallback"
        reranked.append(item)

    return reranked


def rerank_results(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    strategy: str = "hybrid",
    provider_api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank retrieval candidates for final answer synthesis.

    Args:
        query: User query.
        candidates: Retrieved FAISS candidates.
        top_k: Number of reranked results to return.
        strategy: One of "hybrid", "openai_llm", or "anthropic_llm".
        provider_api_key: API key for LLM-based reranking.
        model: LLM model name for LLM-based reranking.
    """
    if not candidates:
        return []

    top_k = max(1, min(top_k, len(candidates)))

    if strategy == "hybrid":
        reranked = []
        for candidate in candidates:
            item = dict(candidate)
            item["vector_score"] = _safe_float(candidate.get("score"))
            item["lexical_score"] = round(
                _lexical_overlap_score(query, str(candidate.get("text") or "")), 6
            )
            item["metadata_score"] = round(
                _metadata_alignment_score(query, candidate), 6
            )
            item["rerank_score"] = round(_hybrid_score(query, candidate), 6)
            item["rerank_strategy"] = "hybrid"
            reranked.append(item)

        reranked.sort(
            key=lambda item: (
                item["rerank_score"],
                item["vector_score"],
                item["lexical_score"],
            ),
            reverse=True,
        )
        logger.info(
            f"Hybrid reranking complete — {len(candidates)} candidates → top {top_k}"
        )
        return reranked[:top_k]

    if strategy == "openai_llm":
        if not provider_api_key:
            raise ValueError("provider_api_key is required for OpenAI reranking")
        reranked = _rerank_with_openai(
            query=query,
            candidates=candidates,
            api_key=provider_api_key,
            model=model or "gpt-4.1-mini",
        )
        logger.info(f"OpenAI reranking complete — {len(candidates)} candidates")
        return reranked[:top_k]

    if strategy == "anthropic_llm":
        if not provider_api_key:
            raise ValueError("provider_api_key is required for Anthropic reranking")
        reranked = _rerank_with_anthropic(
            query=query,
            candidates=candidates,
            api_key=provider_api_key,
            model=model or "claude-3-5-haiku-latest",
        )
        logger.info(f"Anthropic reranking complete — {len(candidates)} candidates")
        return reranked[:top_k]

    raise ValueError(
        "Unknown reranking strategy. Expected one of: "
        "'hybrid', 'openai_llm', 'anthropic_llm'"
    )
