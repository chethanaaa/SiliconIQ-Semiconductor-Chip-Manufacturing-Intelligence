"""
Mixture of Experts (MoE) Router
---------------------------------
Routes each query to the best LLM expert based on task type.

Routing table
─────────────────────────────────────────────────────────────────────
 Expert type          │ Provider    │ Model                  │ Why
──────────────────────┼─────────────┼────────────────────────┼──────────────────────────────
 deep_reasoning       │ Anthropic   │ claude-sonnet-4-5      │ Multi-step technical analysis,
                      │             │                        │ supply chain inference
──────────────────────┼─────────────┼────────────────────────┼──────────────────────────────
 synthesis_narration  │ OpenAI      │ gpt-4o                 │ Fluent, structured narrative
                      │             │                        │ answers, citations, summaries
──────────────────────┼─────────────┼────────────────────────┼──────────────────────────────
 market_analysis      │ Anthropic   │ claude-sonnet-4-5      │ Economic reasoning, trend
                      │             │                        │ interpretation, forecasts
──────────────────────┼─────────────┼────────────────────────┼──────────────────────────────
 data_interpretation  │ OpenAI      │ gpt-4o                 │ Structured data → insight
                      │             │                        │ (FRED series, tables)
──────────────────────┼─────────────┼────────────────────────┼──────────────────────────────
 quick_factual        │ OpenAI      │ gpt-4o-mini            │ Fast, low-cost factual lookups
──────────────────────┼─────────────┼────────────────────────┼──────────────────────────────
 safety_check         │ Anthropic   │ claude-haiku-4-5       │ Fast guardrail evaluation
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import re
from typing import Dict

from loguru import logger


# ── Routing table ─────────────────────────────────────────────────────────────

EXPERT_CONFIGS: Dict[str, Dict[str, str]] = {
    "deep_reasoning": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "temperature": "0.2",
        "description": "Multi-step technical + causal reasoning over supply chain data",
    },
    "synthesis_narration": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "temperature": "0.4",
        "description": "Fluent narrative synthesis with inline citations",
    },
    "market_analysis": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "temperature": "0.2",
        "description": "Economic trend interpretation and forecasting",
    },
    "data_interpretation": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "temperature": "0.3",
        "description": "Converting FRED time-series / tables into insights",
    },
    "quick_factual": {
        "provider": "openai",
        "model": "gpt-4.1-nano",
        "temperature": "0.0",
        "description": "Fast, low-cost factual lookups and definitions",
    },
    "safety_check": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "temperature": "0.0",
        "description": "Guardrail evaluation — fast and conservative",
    },
}

# ── Keyword-based classification signals ─────────────────────────────────────

_REASONING_SIGNALS = [
    r"\bwhy\b", r"\bhow does\b", r"\bimpact\b", r"\bcause[sd]?\b",
    r"\bimplication\b", r"\brisk[s]?\b", r"\bvulnerabilit", r"\bstrateg",
    r"\bcompare\b", r"\bcontrast\b", r"\bexplain\b", r"\banalyze\b",
    r"\btrade.?off\b", r"\bdependenc", r"\bchoke.?point\b",
]

_MARKET_SIGNALS = [
    r"\bmarket size\b", r"\brevenue\b", r"\bcagr\b", r"\bgrowth rate\b",
    r"\bforecast\b", r"\boutlook\b", r"\bprojection\b", r"\bshare\b",
    r"\bvaluation\b", r"\bbillion\b", r"\bmillion\b", r"\btrend\b",
]

_DATA_SIGNALS = [
    r"\bfred\b", r"\bindex\b", r"\bseries\b", r"\bquarterly\b",
    r"\bannual\b", r"\bmonthly\b", r"\bstatistic\b", r"\bdata\b",
    r"\bnumber\b", r"\bfigure\b", r"\btable\b", r"\bchart\b",
    r"\bproduction index\b", r"\bppi\b", r"\bcpi\b",
]

_FACTUAL_SIGNALS = [
    r"\bwhat is\b", r"\bwho is\b", r"\bdefine\b", r"\blist\b",
    r"\bname\b", r"\bwhen\b", r"\bwhere\b", r"\bwhich\b",
    r"\bhow many\b", r"\bhow much\b",
]


def _score(text: str, patterns: list) -> int:
    lower = text.lower()
    return sum(1 for p in patterns if re.search(p, lower))


# ── Public API ────────────────────────────────────────────────────────────────

def classify_expert_type(query: str, tools_called: list[str] | None = None) -> str:
    """
    Classify the query into an expert type using keyword signals.

    Args:
        query       : user query
        tools_called: list of tools the planner chose (influences routing)

    Returns:
        One of the keys in EXPERT_CONFIGS.
    """
    tools_called = tools_called or []

    # FRED data in results → data_interpretation
    if "fetch_fred_data" in tools_called:
        score_data = _score(query, _DATA_SIGNALS)
        if score_data >= 1:
            return "data_interpretation"

    reasoning_score = _score(query, _REASONING_SIGNALS)
    market_score    = _score(query, _MARKET_SIGNALS)
    data_score      = _score(query, _DATA_SIGNALS)
    factual_score   = _score(query, _FACTUAL_SIGNALS)

    scores = {
        "deep_reasoning":      reasoning_score * 3,
        "market_analysis":     market_score * 2,
        "data_interpretation": data_score * 2,
        "quick_factual":       factual_score,
    }

    best = max(scores, key=scores.get)

    # If no clear winner, default to synthesis_narration
    if scores[best] == 0:
        best = "synthesis_narration"

    logger.info(
        f"MoE routing: '{query[:60]}' → {best} "
        f"(scores: reasoning={reasoning_score} market={market_score} "
        f"data={data_score} factual={factual_score})"
    )
    return best


def get_expert_config(expert_type: str) -> Dict[str, str]:
    """Return the LLM config dict for a given expert type."""
    config = EXPERT_CONFIGS.get(expert_type, EXPERT_CONFIGS["synthesis_narration"])
    logger.debug(f"Expert config: {expert_type} → {config['provider']}/{config['model']}")
    return config


def get_llm(config: Dict[str, str]):
    """
    Instantiate and return a LangChain chat model from an expert config dict.
    Lazy import to avoid loading both SDKs if only one is used.
    """
    provider    = config["provider"]
    model       = config["model"]
    temperature = float(config.get("temperature", 0.2))

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        import os
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        import os
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
