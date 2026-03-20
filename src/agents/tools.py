"""
Agent Tools
-----------
Three tool calls available to the agentic layer:

  1. rag_retrieval   — hybrid FAISS + BM25 search over ingested PDFs
  2. fetch_news      — NewsAPI search for semiconductor / supply chain news
  3. fetch_fred_data — FRED economic time-series data (supply chain indicators)

Tools are built as LangChain @tool functions via a factory that injects
the loaded indexes and API keys at graph startup — no global state needed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import requests
from langchain_core.tools import tool
from loguru import logger

from src.rag.retrieval import retrieve_and_rerank, build_bm25_index


# ── Tool context (injected at startup) ───────────────────────────────────────

class ToolContext:
    """Holds loaded indexes and credentials shared across all tool calls."""

    def __init__(
        self,
        faiss_index,
        metadata: List[Dict],
        openai_api_key: str,
        news_api_key: str,
        fred_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        top_k_retrieval: int = 20,
        top_k_rerank: int = 5,
    ):
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.bm25_index = build_bm25_index(metadata)
        self.openai_api_key = openai_api_key
        self.news_api_key = news_api_key
        self.fred_api_key = fred_api_key
        self.embedding_model = embedding_model
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank


# ── Tool factory ─────────────────────────────────────────────────────────────

def create_tools(ctx: ToolContext):
    """
    Return the three tool callables bound to the given ToolContext.
    Call this once at graph startup and pass the tools into the graph.
    """

    # ── Tool 1: RAG retrieval ─────────────────────────────────────────────

    @tool
    def rag_retrieval(query: str) -> str:
        """
        Search the semiconductor supply chain document corpus using hybrid
        dense (FAISS) + sparse (BM25) retrieval.

        Use this for questions about:
        - Industry reports (SIA, McKinsey)
        - Supply chain structure, risks, shortages
        - Chip manufacturing, process nodes, packaging
        - Market size, revenue, CAGR forecasts
        - Geopolitical trade impacts on semiconductors

        Args:
            query: the specific question or sub-question to retrieve docs for

        Returns:
            JSON string with reranked chunks including text, source, page, section.
        """
        logger.info(f"[RAG] query='{query[:80]}'")
        try:
            results = retrieve_and_rerank(
                query=query,
                faiss_index=ctx.faiss_index,
                bm25_index=ctx.bm25_index,
                metadata=ctx.metadata,
                openai_api_key=ctx.openai_api_key,
                embedding_model=ctx.embedding_model,
                top_k_retrieval=ctx.top_k_retrieval,
                top_k_rerank=ctx.top_k_rerank,
            )
            # Keep all citation-relevant fields; trim only embeddings
            slim = [
                {
                    "chunk_id":        r.get("chunk_id", ""),
                    "source_file":     r.get("source_file", "unknown"),
                    "page_num":        r.get("page_num", 0),
                    "total_pages":     r.get("total_pages", 0),
                    "section_header":  r.get("section_header"),
                    "section_level":   r.get("section_level"),
                    "topic":           r.get("topic", "general"),
                    "doc_type":        r.get("doc_type", "general"),
                    "domain_tags":     r.get("domain_tags", []),
                    "position_pct":    r.get("position_pct", 0.0),
                    "text":            r.get("text", "")[:1500],
                    "rerank_score":    r.get("rerank_score", 0.0),
                    "rrf_score":       r.get("rrf_score", 0.0),
                    "dense_score":     r.get("dense_score", 0.0),
                    "sparse_score":    r.get("sparse_score", 0.0),
                    "retrieval_method": r.get("retrieval_method", "unknown"),
                }
                for r in results
            ]
            logger.success(f"[RAG] returned {len(slim)} chunks")
            return json.dumps({"results": slim, "count": len(slim)})
        except Exception as e:
            logger.error(f"[RAG] error: {e}")
            return json.dumps({"error": str(e), "results": []})

    # ── Tool 2: News API ──────────────────────────────────────────────────

    @tool
    def fetch_news(query: str, page_size: int = 10) -> str:
        """
        Fetch recent semiconductor industry news articles from NewsAPI.

        Use this for questions about:
        - Recent supply chain disruptions or shortages
        - Company announcements (TSMC, Intel, NVIDIA, etc.)
        - New fab investments or policy changes
        - Trade restrictions or export controls

        Args:
            query    : news search query (e.g. "TSMC fab expansion Taiwan 2024")
            page_size: number of articles to return (max 20)

        Returns:
            JSON string with articles — title, source, date, description, URL.
        """
        logger.info(f"[NEWS] query='{query[:80]}'")
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "apiKey": ctx.news_api_key,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": min(page_size, 20),
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = [
                {
                    "title": a.get("title"),
                    "source": a.get("source", {}).get("name"),
                    "published_at": a.get("publishedAt"),
                    "description": a.get("description"),
                    "url": a.get("url"),
                }
                for a in data.get("articles", [])
                if a.get("title") and "[Removed]" not in a.get("title", "")
            ]
            logger.success(f"[NEWS] returned {len(articles)} articles")
            return json.dumps({"articles": articles, "count": len(articles)})
        except Exception as e:
            logger.error(f"[NEWS] error: {e}")
            return json.dumps({"error": str(e), "articles": []})

    # ── Tool 3: FRED API ──────────────────────────────────────────────────

    # Curated FRED series relevant to semiconductor supply chain
    _FRED_SERIES: Dict[str, str] = {
        "industrial_production_semiconductors": "IPG3344S",   # semiconductor output index
        "industrial_production_computers":      "IPCONGD",    # computer & electronics
        "ppi_semiconductors":                   "PCU334413334413",  # producer price index
        "us_imports_semiconductors":            "IMPJB",      # goods imports proxy
        "manufacturing_capacity_utilization":   "MCUMFNS",    # manufacturing utilisation
        "us_gdp":                               "GDP",
        "cpi_electronics":                      "CUSR0000SAE1",
    }

    @tool
    def fetch_fred_data(series_key: str, observation_start: str = "2020-01-01") -> str:
        """
        Fetch economic time-series data from the FRED (Federal Reserve) database.

        Use this for questions about:
        - Semiconductor production indices
        - Producer price trends for chips
        - Manufacturing capacity utilisation
        - Import/export volumes
        - Macroeconomic context (GDP, CPI)

        Available series_key values:
            industrial_production_semiconductors
            industrial_production_computers
            ppi_semiconductors
            us_imports_semiconductors
            manufacturing_capacity_utilization
            us_gdp
            cpi_electronics

        Args:
            series_key       : one of the keys listed above
            observation_start: start date in YYYY-MM-DD format (default 2020-01-01)

        Returns:
            JSON string with series metadata and recent observations.
        """
        logger.info(f"[FRED] series='{series_key}' from {observation_start}")

        series_id = _FRED_SERIES.get(series_key)
        if not series_id:
            return json.dumps({
                "error": f"Unknown series_key '{series_key}'. "
                         f"Valid keys: {list(_FRED_SERIES.keys())}",
            })

        try:
            base = "https://api.stlouisfed.org/fred"
            # Fetch series metadata
            meta_resp = requests.get(
                f"{base}/series",
                params={"series_id": series_id, "api_key": ctx.fred_api_key, "file_type": "json"},
                timeout=10,
            )
            meta_resp.raise_for_status()
            meta = meta_resp.json().get("seriess", [{}])[0]

            # Fetch observations
            obs_resp = requests.get(
                f"{base}/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": ctx.fred_api_key,
                    "file_type": "json",
                    "observation_start": observation_start,
                    "sort_order": "desc",
                    "limit": 24,       # last 24 observations (~2 years monthly)
                },
                timeout=10,
            )
            obs_resp.raise_for_status()
            observations = obs_resp.json().get("observations", [])

            result = {
                "series_id": series_id,
                "title": meta.get("title"),
                "units": meta.get("units"),
                "frequency": meta.get("frequency"),
                "last_updated": meta.get("last_updated"),
                "observations": [
                    {"date": o["date"], "value": o["value"]}
                    for o in observations
                    if o.get("value") != "."    # FRED uses "." for missing
                ],
            }
            logger.success(
                f"[FRED] {series_id} — {len(result['observations'])} observations"
            )
            return json.dumps(result)
        except Exception as e:
            logger.error(f"[FRED] error: {e}")
            return json.dumps({"error": str(e)})

    return {
        "rag_retrieval": rag_retrieval,
        "fetch_news": fetch_news,
        "fetch_fred_data": fetch_fred_data,
    }
