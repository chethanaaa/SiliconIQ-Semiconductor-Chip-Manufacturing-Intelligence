"""
LangGraph Graph
---------------
Assembles all nodes into the agentic RAG graph with persistent memory.

Graph topology:
                          ┌─────────────┐
               START ───► │   planner   │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │tool_executor│
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │ moe_router  │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │ synthesizer │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │   safety    │
                          └──────┬──────┘
                         pass /      \ fail
                              │        │
                            END     planner (retry)

Persistence:
    MemorySaver with thread_id gives full conversation memory per session.
    Each turn appends to state["messages"] — the graph remembers all prior Q&A.
"""

from __future__ import annotations

import functools
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from src.agents.nodes import (
    moe_router_node,
    planner_node,
    safety_node,
    synthesizer_node,
    tool_executor_node,
)
from src.agents.state import AgentState
from src.agents.tools import ToolContext, create_tools
from src.rag.vector_store import load_index


# ── Conditional edge: after safety ───────────────────────────────────────────

def _safety_router(state: AgentState) -> str:
    """Safety is a warning system — always proceed to END, never retry."""
    logger.info("[GRAPH] safety complete → END")
    return END


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph(
    faiss_index_path: str | None = None,
    metadata_path: str | None = None,
):
    """
    Build and compile the agentic RAG LangGraph.

    Loads the FAISS index from disk (must have been built by run_ingestion_test.py
    or the ingestion pipeline), creates all tools, wires up nodes, and compiles
    with MemorySaver for persistent conversation memory.

    Args:
        faiss_index_path: path to faiss_index.bin (defaults to .env FAISS_INDEX_PATH)
        metadata_path   : path to metadata.json (defaults to .env FAISS_METADATA_PATH)

    Returns:
        Compiled LangGraph app ready for .invoke() / .stream()
    """
    # ── Load indexes ─────────────────────────────────────────────────────────
    index_path = faiss_index_path or os.getenv("FAISS_INDEX_PATH", "data/vector_store/faiss_index.bin")
    meta_path  = metadata_path   or os.getenv("FAISS_METADATA_PATH", "data/vector_store/metadata.json")

    logger.info(f"Loading FAISS index from {index_path}")
    faiss_index, metadata = load_index(index_path, meta_path)

    # ── Build tool context + tools ───────────────────────────────────────────
    ctx = ToolContext(
        faiss_index=faiss_index,
        metadata=metadata,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        news_api_key=os.getenv("NEWS_API_KEY"),
        fred_api_key=os.getenv("FRED_API_KEY"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", 20)),
        top_k_rerank=int(os.getenv("TOP_K_RERANK", 5)),
    )
    tools = create_tools(ctx)
    logger.info(f"Tools loaded: {list(tools.keys())}")

    # ── Wire nodes (partial-bind tools where needed) ─────────────────────────
    graph = StateGraph(AgentState)

    graph.add_node("planner",       functools.partial(planner_node,       tools=tools))
    graph.add_node("tool_executor", functools.partial(tool_executor_node, tools=tools))
    graph.add_node("moe_router",    moe_router_node)
    graph.add_node("synthesizer",   synthesizer_node)
    graph.add_node("safety",        safety_node)

    # ── Edges ────────────────────────────────────────────────────────────────
    graph.add_edge(START,          "planner")
    graph.add_edge("planner",      "tool_executor")
    graph.add_edge("tool_executor","moe_router")
    graph.add_edge("moe_router",   "synthesizer")
    graph.add_edge("synthesizer",  "safety")

    # Safety node: pass → END, fail → planner (retry)
    graph.add_conditional_edges("safety", _safety_router)

    # ── Compile with persistent memory ───────────────────────────────────────
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    logger.success("LangGraph compiled — agentic RAG ready")
    return app


# ── Shared initial state builder ─────────────────────────────────────────────

def _build_initial_state(query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "messages": [],
        "plan": {},
        "tools_to_call": [],
        "tool_results": {},
        "context_chunks": [],
        "expert_type": "",
        "expert_llm_config": {},
        "draft_answer": "",
        "citations": [],
        "safety_passed": False,
        "safety_feedback": "",
        "final_answer": "",
        "iteration": 0,
    }


# ── Convenience runner ────────────────────────────────────────────────────────

def run_query(
    app,
    query: str,
    thread_id: str = "default",
) -> Dict[str, Any]:
    """
    Run a single query through the compiled graph.

    Args:
        app      : compiled LangGraph app from build_graph()
        query    : user question
        thread_id: session identifier — same thread_id preserves conversation memory

    Returns:
        Dict with keys: final_answer, citations, expert_type, tools_called, plan
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = _build_initial_state(query)

    logger.info(f"[GRAPH] query='{query[:80]}' thread={thread_id}")
    start_time = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    final_state: Dict[str, Any] | None = None
    error_message = None

    try:
        final_state = app.invoke(initial_state, config=config)
    except Exception as exc:
        error_message = str(exc)
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        latency_ms = (time.perf_counter() - t0) * 1000
        record_query_latency(
            query=query,
            thread_id=thread_id,
            latency_ms=latency_ms,
            start_time=start_time,
            end_time=end_time,
            success=error_message is None,
            error=error_message,
            extra_metadata={
                "has_final_state": final_state is not None,
            },
        )
        logger.info(f"[GRAPH] latency_ms={latency_ms:.2f} thread={thread_id}")

    return {
        "final_answer": final_state.get("final_answer", ""),
        "citations":    final_state.get("citations", []),
        "expert_type":  final_state.get("expert_type", ""),
        "tools_called": final_state.get("tools_to_call", []),
        "plan":         final_state.get("plan", {}),
    }
