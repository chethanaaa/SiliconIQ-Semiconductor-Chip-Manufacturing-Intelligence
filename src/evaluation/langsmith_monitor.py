"""
LangSmith Monitoring
--------------------
Minimal monitoring helpers for recording one latency event per user query.

This intentionally tracks only top-level query latency for now rather than
tracing every internal tool/agent step.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from langsmith import Client
from loguru import logger


def _is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_langsmith_monitoring_enabled() -> bool:
    """Return True when latency monitoring should be sent to LangSmith."""
    if _is_truthy(os.getenv("LANGSMITH_MONITORING_ENABLED")):
        return True
    return _is_truthy(os.getenv("LANGSMITH_TRACING"))


def _build_client() -> Client:
    api_key = os.getenv("LANGSMITH_API_KEY")
    endpoint = os.getenv("LANGSMITH_ENDPOINT")
    return Client(
        api_key=api_key,
        api_url=endpoint or None,
        auto_batch_tracing=False,
    )


def record_query_latency(
    query: str,
    thread_id: str,
    latency_ms: float,
    start_time: datetime,
    end_time: datetime,
    success: bool,
    error: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Record a single user-query latency run in LangSmith.

    Returns:
        True if a LangSmith run was submitted, otherwise False.
    """
    if not is_langsmith_monitoring_enabled():
        return False

    if not os.getenv("LANGSMITH_API_KEY"):
        logger.warning("LangSmith monitoring enabled, but LANGSMITH_API_KEY is missing")
        return False

    project_name = os.getenv("LANGSMITH_PROJECT", "manufacturing-agentic-rag")
    run_id = uuid4()
    metadata = {
        "thread_id": thread_id,
        "latency_ms": round(latency_ms, 2),
        "success": success,
        "monitoring": "latency_only",
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    try:
        client = _build_client()
        client.create_run(
            id=run_id,
            project_name=project_name,
            name="user_query_latency",
            run_type="chain",
            inputs={
                "query": query,
                "thread_id": thread_id,
            },
            outputs={
                "latency_ms": round(latency_ms, 2),
                "success": success,
            },
            start_time=start_time,
            end_time=end_time,
            error=error,
            extra={"metadata": metadata},
            tags=["latency", "user-query"],
        )
        logger.info(
            f"[LANGSMITH] recorded query latency | thread={thread_id} "
            f"| latency_ms={latency_ms:.2f}"
        )
        return True
    except Exception as exc:
        logger.warning(f"[LANGSMITH] failed to record latency: {exc}")
        return False
