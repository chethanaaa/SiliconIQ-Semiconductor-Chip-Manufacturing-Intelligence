"""
Citation
--------
Attaches fully structured source provenance to every RAG response.

Each answer returned to the user is paired with a CitedResponse that contains:
  - The answer text
  - An ordered list of Citation objects (one per source chunk used)
  - A formatted reference block ready to display in the chat UI

Citation fields per source chunk:
    document     : original PDF filename
    page         : page number (1-based)
    total_pages  : total pages in the source document
    section      : nearest section heading detected in the chunk
    section_level: H1 / H2 / H3
    topic        : inferred semantic topic
    doc_type     : document category
    position_pct : where in the document the chunk sits (0–100 %)
    domain_tags  : semiconductor domain keywords matched
    excerpt      : first 200 chars of the chunk text
    chunk_id     : unique chunk identifier
    rerank_score : final ranking score (higher = more relevant)
    retrieval_method : dense_only | sparse_only | both
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Citation:
    # Provenance
    citation_number: int
    chunk_id: str
    document: str
    page: int
    total_pages: int

    # Structure
    section: Optional[str]
    section_level: Optional[str]
    position_pct: float

    # Semantic classification
    topic: str
    doc_type: str
    domain_tags: List[str]

    # Excerpt shown to user
    excerpt: str

    # Retrieval signals
    rerank_score: float
    rrf_score: float
    dense_score: float
    sparse_score: float
    retrieval_method: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_inline(self) -> str:
        """Short inline reference: [1] — used inside answer text."""
        return f"[{self.citation_number}]"

    def to_reference_block(self) -> str:
        """Full reference block for display below the answer."""
        section_str = (
            f"{self.section} ({self.section_level})"
            if self.section else "—"
        )
        tags = ", ".join(self.domain_tags[:6]) if self.domain_tags else "—"
        lines = [
            f"[{self.citation_number}] {self.document}",
            f"    Page         : {self.page} / {self.total_pages}",
            f"    Section      : {section_str}",
            f"    Position     : {self.position_pct:.1f}% through document",
            f"    Topic        : {self.topic.replace('_', ' ').title()}",
            f"    Doc type     : {self.doc_type.replace('_', ' ').title()}",
            f"    Domain tags  : {tags}",
            f"    Relevance    : {self.rerank_score:.4f}  "
            f"(dense={self.dense_score:.4f} | sparse={self.sparse_score:.4f} | "
            f"method={self.retrieval_method})",
            f"    Excerpt      : \"{self.excerpt}\"",
        ]
        return "\n".join(lines)


@dataclass
class CitedResponse:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    query: str = ""

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def source_documents(self) -> List[str]:
        """Unique document names referenced, in citation order."""
        seen = set()
        docs = []
        for c in self.citations:
            if c.document not in seen:
                seen.add(c.document)
                docs.append(c.document)
        return docs

    @property
    def reference_block(self) -> str:
        """Full formatted reference section, ready for chat display."""
        if not self.citations:
            return "No sources cited."
        header = f"{'─' * 60}\nSources\n{'─' * 60}"
        refs = "\n\n".join(c.to_reference_block() for c in self.citations)
        return f"{header}\n{refs}"

    def to_full_response(self) -> str:
        """Answer + reference block as a single string for the chat interface."""
        return f"{self.answer}\n\n{self.reference_block}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "source_documents": self.source_documents,
        }


# ── Builder ───────────────────────────────────────────────────────────────────

def _safe(val: Any, default: Any) -> Any:
    return val if val is not None else default


def _excerpt(text: str, max_chars: int = 200) -> str:
    """First max_chars of text, collapsed whitespace, ending cleanly."""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    # Trim to last full word
    trimmed = flat[:max_chars].rsplit(" ", 1)[0]
    return trimmed + " …"


def build_citations(reranked_results: List[Dict[str, Any]]) -> List[Citation]:
    """
    Convert reranked chunk dicts into Citation objects.
    Each chunk becomes one numbered citation in order of relevance.
    """
    citations: List[Citation] = []
    for i, chunk in enumerate(reranked_results, start=1):
        citations.append(
            Citation(
                citation_number=i,
                chunk_id=_safe(chunk.get("chunk_id"), f"chunk_{i}"),
                document=_safe(chunk.get("source_file"), "unknown"),
                page=int(_safe(chunk.get("page_num"), 0)),
                total_pages=int(_safe(chunk.get("total_pages"), 0)),
                section=chunk.get("section_header"),
                section_level=chunk.get("section_level"),
                position_pct=float(_safe(chunk.get("position_pct"), 0.0)),
                topic=_safe(chunk.get("topic"), "general"),
                doc_type=_safe(chunk.get("doc_type"), "general"),
                domain_tags=list(_safe(chunk.get("domain_tags"), [])),
                excerpt=_excerpt(_safe(chunk.get("text"), "")),
                rerank_score=float(_safe(chunk.get("rerank_score"), 0.0)),
                rrf_score=float(_safe(chunk.get("rrf_score"), 0.0)),
                dense_score=float(_safe(chunk.get("dense_score"), 0.0)),
                sparse_score=float(_safe(chunk.get("sparse_score"), 0.0)),
                retrieval_method=_safe(chunk.get("retrieval_method"), "unknown"),
            )
        )
    return citations


def attach_citations(
    answer: str,
    reranked_results: List[Dict[str, Any]],
    query: str = "",
) -> CitedResponse:
    """
    Wrap an LLM answer with structured citations from retrieval results.

    Args:
        answer           : the generated answer string from the LLM
        reranked_results : output of retrieve_and_rerank()
        query            : original user query (stored for traceability)

    Returns:
        CitedResponse with answer + full citation metadata
    """
    citations = build_citations(reranked_results)
    return CitedResponse(answer=answer, citations=citations, query=query)


# ── Context builder for LLM prompt ───────────────────────────────────────────

def build_context_block(reranked_results: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.
    Each chunk is prefixed with its citation number so the LLM can reference
    it inline with [1], [2], etc.

    Example output:
        [1] Source: SIA-2025.pdf | Page 12 | Section: Global Revenue Trends
        <chunk text>

        [2] Source: mck_semiconductors_2024.pdf | Page 34 | Section: Supply Risk
        <chunk text>
    """
    blocks: List[str] = []
    for i, chunk in enumerate(reranked_results, start=1):
        source   = _safe(chunk.get("source_file"), "unknown")
        page     = _safe(chunk.get("page_num"), "?")
        section  = chunk.get("section_header") or "—"
        topic    = _safe(chunk.get("topic"), "general").replace("_", " ")
        text     = _safe(chunk.get("text"), "").strip()

        header = (
            f"[{i}] Source: {source} | Page: {page} | "
            f"Section: {section} | Topic: {topic}"
        )
        blocks.append(f"{header}\n{textwrap.fill(text, width=100)}")

    return "\n\n".join(blocks)
