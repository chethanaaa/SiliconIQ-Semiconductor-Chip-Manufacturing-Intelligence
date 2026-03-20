"""
Agent State
-----------
Shared LangGraph state that persists across all nodes in the graph.

MemorySaver (keyed by thread_id) gives us conversation-level persistence —
each user session retains its full history across turns.

Fields
------
query               : current user query
messages            : full conversation history (LangChain BaseMessage list)
plan                : structured plan from the Planner node
tools_to_call       : list of tool names the planner selected
tool_results        : {tool_name: result_dict} from the tool executor
context_chunks      : reranked RAG chunks ready for synthesis
expert_type         : task type chosen by the MoE router
expert_llm_config   : {provider, model} selected by MoE for synthesis
draft_answer        : raw answer from the Synthesizer
citations           : list of citation dicts attached to the answer
safety_passed       : True if safety node approved the answer
safety_feedback     : reason if safety node rejected
final_answer        : approved answer returned to the user
iteration           : guard against infinite loops (max 3 retries)
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Query ────────────────────────────────────────────────────────────────
    query: str

    # ── Conversation memory (append-only via add_messages reducer) ───────────
    messages: Annotated[List[BaseMessage], add_messages]

    # ── Planner output ───────────────────────────────────────────────────────
    plan: Dict[str, Any]               # full structured plan
    tools_to_call: List[str]           # e.g. ["rag_retrieval", "fetch_news"]

    # ── Tool results ─────────────────────────────────────────────────────────
    tool_results: Dict[str, Any]       # {tool_name: result}
    context_chunks: List[Dict]         # reranked RAG chunks

    # ── MoE routing ──────────────────────────────────────────────────────────
    expert_type: str                   # e.g. "deep_reasoning"
    expert_llm_config: Dict[str, str]  # {provider, model, temperature}

    # ── Answer lifecycle ─────────────────────────────────────────────────────
    draft_answer: str
    citations: List[Dict]
    safety_passed: bool
    safety_feedback: str
    final_answer: str

    # ── Loop guard ───────────────────────────────────────────────────────────
    iteration: int
