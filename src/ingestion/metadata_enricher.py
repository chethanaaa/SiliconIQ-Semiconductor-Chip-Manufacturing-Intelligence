"""
Metadata Enricher
-----------------
Enriches each TextChunk with structured metadata:

  - section_header   : nearest detected section heading above the chunk
  - section_level    : H1 / H2 / H3
  - topic            : inferred high-level topic from a semiconductor taxonomy
  - doc_type         : document category (industry_report, supply_chain, etc.)
  - domain_tags      : semiconductor-domain keywords found in the chunk
  - word_count       : number of words
  - has_table        : heuristic flag for tabular content
  - industry         : fixed label "semiconductor_supply_chain"
  - position_pct     : chunk position as % through the document (0–100)
"""

import re
from typing import List, Optional, Tuple

from loguru import logger

from src.ingestion.chunker import TextChunk


# ── Section header detection ─────────────────────────────────────────────────

# Numbered sections:  "1.", "2.1", "3.1.2" followed by a heading word
_NUMBERED_SECTION = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2}){0,2})\s+([A-Z][^\n]{3,80})$", re.MULTILINE
)

# ALL-CAPS headings (at least 3 words, not a sentence — no period at end)
_ALLCAPS_HEADING = re.compile(
    r"^([A-Z][A-Z\s\-&:]{8,60})$", re.MULTILINE
)

# Title Case headings (3–10 words, no trailing period)
_TITLECASE_HEADING = re.compile(
    r"^((?:[A-Z][a-z]+[\s\-]){2,9}[A-Z][a-z]+)$", re.MULTILINE
)


def _detect_section(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (section_header, section_level) for the first heading found in text.
    Priority: numbered > all-caps > title-case
    """
    m = _NUMBERED_SECTION.search(text)
    if m:
        num = m.group(1)
        depth = num.count(".") + 1          # "2.1" → depth 2
        level = f"H{min(depth, 3)}"
        return m.group(2).strip(), level

    m = _ALLCAPS_HEADING.search(text)
    if m:
        return m.group(1).strip(), "H1"

    m = _TITLECASE_HEADING.search(text)
    if m:
        return m.group(1).strip(), "H2"

    return None, None


# ── Topic taxonomy ───────────────────────────────────────────────────────────

_TOPIC_KEYWORDS: dict[str, List[str]] = {
    "market_size_and_growth": [
        "market size", "revenue", "cagr", "forecast", "growth rate",
        "billion", "market share", "outlook", "projection",
    ],
    "supply_chain_risk": [
        "supply chain", "shortage", "disruption", "lead time", "bottleneck",
        "single source", "concentration", "geopolitical", "risk",
    ],
    "chip_manufacturing_process": [
        "wafer", "fab", "fabrication", "process node", "nm", "lithography",
        "euv", "duv", "deposition", "etching", "cmp", "yield", "foundry",
    ],
    "chip_design": [
        "asic", "fpga", "soc", "ip core", "eda", "rtl", "chip design",
        "fabless", "architecture", "tape-out", "verification",
    ],
    "advanced_packaging": [
        "packaging", "osat", "chiplet", "advanced packaging", "2.5d", "3d",
        "hbm", "interposer", "substrate", "flip chip",
    ],
    "geopolitics_and_trade": [
        "export control", "bis", "entity list", "tariff", "trade war",
        "sanction", "chips act", "reshoring", "china", "taiwan",
    ],
    "raw_materials": [
        "raw material", "silicon", "rare earth", "chemical", "gas",
        "photoresist", "neon", "palladium", "cobalt",
    ],
    "company_and_ecosystem": [
        "tsmc", "samsung", "intel", "nvidia", "amd", "qualcomm",
        "apple", "broadcom", "asml", "applied materials", "lam research",
        "kla", "sia", "semi",
    ],
    "investment_and_policy": [
        "investment", "subsidy", "government", "policy", "chips act",
        "funding", "capex", "r&d", "incentive",
    ],
}


def _infer_topic(text: str) -> str:
    lower = text.lower()
    scores: dict[str, int] = {}
    for topic, keywords in _TOPIC_KEYWORDS.items():
        score = sum(lower.count(kw) for kw in keywords)
        if score > 0:
            scores[topic] = score
    return max(scores, key=scores.get) if scores else "general"


# ── Doc type inference ───────────────────────────────────────────────────────

_DOC_TYPE_PATTERNS: dict[str, List[str]] = {
    "industry_report": [
        r"\bstate of the industry\b", r"\bannual report\b",
        r"\bmarket (report|outlook|forecast)\b", r"\bsia\b",
    ],
    "supply_chain_brief": [
        r"\bsupply chain\b", r"\bsupplier\b", r"\bprocurement\b",
        r"\bsourcing\b", r"\blogistics\b",
    ],
    "technical_analysis": [
        r"\bfabrication\b", r"\bwafer\b", r"\bphotolithography\b",
        r"\bprocess node\b", r"\beuv\b",
    ],
    "economic_analysis": [
        r"\bgdp\b", r"\bmarket size\b", r"\brevenue\b",
        r"\bcagr\b", r"\bforecast\b",
    ],
    "geopolitical_analysis": [
        r"\btariff\b", r"\bexport control\b", r"\bsanction\b",
        r"\bchips act\b", r"\bgeopolit\b",
    ],
}


def _infer_doc_type(filename: str, sample_text: str) -> str:
    combined = (filename + " " + sample_text).lower()
    scores: dict[str, int] = {}
    for doc_type, patterns in _DOC_TYPE_PATTERNS.items():
        score = sum(len(re.findall(p, combined)) for p in patterns)
        if score > 0:
            scores[doc_type] = score
    return max(scores, key=scores.get) if scores else "general"


# ── Domain keyword tagging ───────────────────────────────────────────────────

_DOMAIN_KEYWORDS = [
    "ASIC", "FPGA", "SoC", "EDA", "RTL", "IP core", "chip design",
    "fabless", "foundry", "TSMC", "Samsung", "Intel", "ASML",
    "wafer", "fab", "fabrication", "process node", "lithography",
    "EUV", "DUV", "photomask", "deposition", "etching", "CMP", "yield",
    "packaging", "OSAT", "chiplet", "advanced packaging", "HBM",
    "supply chain", "shortage", "lead time", "inventory", "procurement",
    "silicon", "rare earth", "substrate",
    "NVIDIA", "AMD", "Qualcomm", "Apple", "Broadcom", "MediaTek",
    "semiconductor", "SIA", "SEMI", "CAGR",
    "export control", "BIS", "tariff", "CHIPS Act", "reshoring",
    "China", "Taiwan", "Korea",
]

_KW_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _DOMAIN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _extract_domain_tags(text: str) -> List[str]:
    matches = _KW_PATTERN.findall(text)
    seen: set = set()
    tags = []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            tags.append(m)
    return tags


def _has_table(text: str) -> bool:
    return bool(re.search(r"\t.+\t|\|.+\|.+\|", text))


# ── Doc-level context cache ──────────────────────────────────────────────────

def _build_doc_type_cache(chunks: List[TextChunk]) -> dict[str, str]:
    """Infer doc_type once per source file using the first 3 chunks as sample."""
    cache: dict[str, str] = {}
    from collections import defaultdict
    file_samples: dict[str, List[str]] = defaultdict(list)
    for c in chunks:
        if len(file_samples[c.source_file]) < 3:
            file_samples[c.source_file].append(c.text)
    for filename, samples in file_samples.items():
        cache[filename] = _infer_doc_type(filename, " ".join(samples))
    return cache


# ── Public API ───────────────────────────────────────────────────────────────

def enrich_chunk(
    chunk: TextChunk,
    doc_type: str,
    total_chunks_in_doc: int,
) -> TextChunk:
    """Enrich a single chunk with all metadata fields."""
    section_header, section_level = _detect_section(chunk.text)

    # Position as percentage through the document
    position_pct = round(
        (chunk.chunk_index / max(total_chunks_in_doc - 1, 1)) * 100, 1
    )

    chunk.metadata.update(
        {
            # Provenance
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "page_num": chunk.page_num,
            "total_pages": chunk.total_pages,
            "chunk_index": chunk.chunk_index,
            "position_pct": position_pct,
            # Content structure
            "section_header": section_header,
            "section_level": section_level,
            "has_table": _has_table(chunk.text),
            # Counts
            "char_count": chunk.char_count,
            "word_count": len(chunk.text.split()),
            # Semantic classification
            "industry": "semiconductor_supply_chain",
            "doc_type": doc_type,
            "topic": _infer_topic(chunk.text),
            "domain_tags": _extract_domain_tags(chunk.text),
        }
    )
    return chunk


def enrich_chunks(chunks: List[TextChunk]) -> List[TextChunk]:
    """Enrich all chunks. Doc type is inferred once per source file."""
    doc_type_cache = _build_doc_type_cache(chunks)

    # Count chunks per source file for position_pct
    from collections import Counter
    file_chunk_counts = Counter(c.source_file for c in chunks)

    enriched = [
        enrich_chunk(
            chunk,
            doc_type=doc_type_cache[chunk.source_file],
            total_chunks_in_doc=file_chunk_counts[chunk.source_file],
        )
        for chunk in chunks
    ]

    topics = [c.metadata["topic"] for c in enriched]
    from collections import Counter as C
    top_topics = C(topics).most_common(3)
    logger.success(
        f"Enriched {len(enriched)} chunks | "
        f"top topics: {', '.join(f'{t}({n})' for t, n in top_topics)}"
    )
    return enriched
