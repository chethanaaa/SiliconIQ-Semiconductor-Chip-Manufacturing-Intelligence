"""
LangGraph Nodes
---------------
Each function here is one node in the LangGraph graph.

Node execution order:
  planner → tool_executor → moe_router → synthesizer → safety → END
                                                              ↓ (fail)
                                                          planner (retry)

Nodes receive the full AgentState and return a partial dict of state updates.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from src.agents.crew_agents import run_domain_agents, run_planner, run_synthesizer
from src.agents.moe_router import classify_expert_type, get_expert_config, get_llm
from src.agents.state import AgentState
from src.rag.citation import attach_citations, build_context_block

# Max retries before giving up
_MAX_ITERATIONS = 3


# ── Node 1: Planner ───────────────────────────────────────────────────────────

def planner_node(state: AgentState, tools: Dict[str, Any]) -> Dict:
    """
    Analyzes the query, produces a research plan via CrewAI Planner agent,
    and determines which tools to invoke.
    """
    query = state["query"]
    iteration = state.get("iteration", 0)
    safety_feedback = state.get("safety_feedback", "")

    logger.info(f"[PLANNER] iteration={iteration} query='{query[:80]}'")

    # Build query with safety feedback context if this is a retry
    enriched_query = query
    if safety_feedback:
        enriched_query = (
            f"{query}\n\n[Revision note: previous answer was flagged — {safety_feedback}]"
        )

    # Planner uses Claude Sonnet for deep reasoning
    from src.agents.moe_router import EXPERT_CONFIGS
    planner_llm = get_llm(EXPERT_CONFIGS["deep_reasoning"])
    plan = run_planner(enriched_query, planner_llm)

    tools_to_call = plan.get("tools_to_call", ["rag_retrieval"])
    # Validate tool names
    valid_tools = set(tools.keys())
    tools_to_call = [t for t in tools_to_call if t in valid_tools]
    if not tools_to_call:
        tools_to_call = ["rag_retrieval"]

    # Validate agent names
    valid_agents = {"procurement", "risk", "manufacturing"}
    agents_to_call = [a for a in plan.get("agents_to_call", []) if a in valid_agents]
    plan["agents_to_call"] = agents_to_call

    return {
        "plan": plan,
        "tools_to_call": tools_to_call,
        "iteration": iteration + 1,
        "messages": [HumanMessage(content=query)],
    }


# ── Node 2: Tool Executor ─────────────────────────────────────────────────────

def tool_executor_node(state: AgentState, tools: Dict[str, Any]) -> Dict:
    """
    Executes the tools selected by the planner in parallel-friendly order.
    Results are stored in state["tool_results"] and RAG chunks in state["context_chunks"].
    """
    plan = state["plan"]
    tools_to_call = state["tools_to_call"]

    logger.info(f"[TOOLS] executing: {tools_to_call}")

    tool_results: Dict[str, Any] = {}
    context_chunks = []

    for tool_name in tools_to_call:
        tool_fn = tools.get(tool_name)
        if not tool_fn:
            logger.warning(f"[TOOLS] unknown tool: {tool_name}")
            continue

        try:
            if tool_name == "rag_retrieval":
                query = plan.get("rag_query") or state["query"]
                raw = tool_fn.invoke({"query": query})

            elif tool_name == "fetch_news":
                query = plan.get("news_query") or state["query"]
                raw = tool_fn.invoke({"query": query})

            elif tool_name == "fetch_fred_data":
                series_key = plan.get("fred_series_key", "industrial_production_semiconductors")
                raw = tool_fn.invoke({"series_key": series_key})

            else:
                raw = tool_fn.invoke({"query": state["query"]})

            result = json.loads(raw) if isinstance(raw, str) else raw
            tool_results[tool_name] = result

            # Pull RAG chunks into dedicated state field
            if tool_name == "rag_retrieval":
                context_chunks = result.get("results", [])

            logger.success(f"[TOOLS] {tool_name} completed")

        except Exception as e:
            logger.error(f"[TOOLS] {tool_name} failed: {e}")
            tool_results[tool_name] = {"error": str(e)}

    return {
        "tool_results": tool_results,
        "context_chunks": context_chunks,
    }


# ── Node 3: MoE Router ────────────────────────────────────────────────────────

def moe_router_node(state: AgentState) -> Dict:
    """
    Classifies the query into an expert type and selects the appropriate LLM
    for the synthesizer node.
    """
    query = state["query"]
    tools_called = state.get("tools_to_call", [])

    expert_type = classify_expert_type(query, tools_called)
    expert_config = get_expert_config(expert_type)

    logger.info(
        f"[MOE] expert_type={expert_type} → "
        f"{expert_config['provider']}/{expert_config['model']}"
    )

    return {
        "expert_type": expert_type,
        "expert_llm_config": expert_config,
        "messages": [
            AIMessage(
                content=f"[MoE] Routing to {expert_type} expert "
                        f"({expert_config['provider']}/{expert_config['model']})"
            )
        ],
    }


# ── Node 4: Synthesizer ───────────────────────────────────────────────────────

def synthesizer_node(state: AgentState) -> Dict:
    """
    Assembles the final answer via CrewAI Synthesizer agent using the LLM
    selected by the MoE router.

    Builds a numbered context block from:
      - RAG chunks (if retrieved)
      - News articles (formatted as sources)
      - FRED data (formatted as data context)
    """
    query = state["query"]
    plan = state["plan"]
    tool_results = state.get("tool_results", {})
    context_chunks = state.get("context_chunks", [])
    expert_type = state.get("expert_type", "synthesis_narration")
    expert_config = state.get("expert_llm_config", {})

    logger.info(f"[SYNTHESIZER] task_type={expert_type}")

    # ── Build context block ─────────────────────────────────────────────────
    context_parts = []

    # RAG context
    if context_chunks:
        context_parts.append(build_context_block(context_chunks))

    # News context
    news_result = tool_results.get("fetch_news", {})
    articles = news_result.get("articles", [])
    if articles:
        news_lines = ["\n── RECENT NEWS ─────────────────────────────"]
        for i, a in enumerate(articles[:5], start=len(context_chunks) + 1):
            news_lines.append(
                f"[{i}] {a.get('title')} | {a.get('source')} | {a.get('published_at', '')[:10]}\n"
                f"    {a.get('description', '')}"
            )
        context_parts.append("\n".join(news_lines))

    # FRED data context
    fred_result = tool_results.get("fetch_fred_data", {})
    if fred_result and "observations" in fred_result:
        obs = fred_result["observations"][:12]
        fred_lines = [
            f"\n── FRED DATA: {fred_result.get('title')} ({fred_result.get('units')}) ─────"
        ]
        for o in obs:
            fred_lines.append(f"    {o['date']}: {o['value']}")
        context_parts.append("\n".join(fred_lines))

    context_block = "\n\n".join(context_parts) if context_parts else "No context retrieved."

    # ── Run domain agents (Procurement / Risk / Manufacturing) ───────────────
    agents_to_call = plan.get("agents_to_call", [])
    from src.agents.moe_router import EXPERT_CONFIGS
    domain_outputs = run_domain_agents(
        query=query,
        core_question=plan.get("core_question", query),
        context_block=context_block,
        agents_to_call=agents_to_call,
        procurement_llm=get_llm(EXPERT_CONFIGS["synthesis_narration"]),   # GPT-4o
        risk_llm=get_llm(EXPERT_CONFIGS["deep_reasoning"]),               # Claude Sonnet
        manufacturing_llm=get_llm(EXPERT_CONFIGS["deep_reasoning"]),      # Claude Sonnet
    )

    # ── Run synthesizer ──────────────────────────────────────────────────────
    synthesizer_llm = get_llm(expert_config) if expert_config else get_llm(
        get_expert_config("synthesis_narration")
    )

    draft = run_synthesizer(
        query=query,
        core_question=plan.get("core_question", query),
        context_block=context_block,
        domain_outputs=domain_outputs,
        task_type=expert_type,
        llm=synthesizer_llm,
    )

    # ── Attach RAG citations ─────────────────────────────────────────────────
    cited = attach_citations(
        answer=draft,
        reranked_results=context_chunks,
        query=query,
    )
    all_citations = [c.to_dict() for c in cited.citations]
    n = len(all_citations) + 1

    # ── Append News citations ─────────────────────────────────────────────────
    news_result = tool_results.get("fetch_news", {})
    for a in news_result.get("articles", [])[:5]:
        all_citations.append({
            "citation_number": n,
            "source_type": "news",
            "document": a.get("source", "News"),
            "title": a.get("title", ""),
            "published_at": a.get("published_at", "")[:10],
            "excerpt": a.get("description", ""),
            "url": a.get("url", ""),
            "topic": "news",
            "domain_tags": [],
        })
        n += 1

    # ── Append FRED citations ─────────────────────────────────────────────────
    fred_result = tool_results.get("fetch_fred_data", {})
    if fred_result and "observations" in fred_result:
        all_citations.append({
            "citation_number": n,
            "source_type": "fred",
            "document": "Federal Reserve (FRED)",
            "title": fred_result.get("title", ""),
            "units": fred_result.get("units", ""),
            "frequency": fred_result.get("frequency", ""),
            "last_updated": fred_result.get("last_updated", ""),
            "topic": "economic_data",
            "domain_tags": [],
            "excerpt": f"{fred_result.get('title')} — {fred_result.get('units')}",
        })

    return {
        "draft_answer": draft,
        "citations": all_citations,
        "messages": [AIMessage(content=draft)],
    }


# ── Node 5: Safety ────────────────────────────────────────────────────────────

_SAFETY_SYSTEM = """
You are a quality reviewer for a semiconductor industry research assistant.
Check only for CRITICAL issues — things that would seriously mislead a reader:

1. INVENTED FACTS : completely fabricated company names, events, or statistics
                    that have no basis in the answer context
2. WRONG DOMAIN   : answer is entirely off-topic (not semiconductor / supply chain)
3. EMPTY ANSWER   : answer has no useful content

Do NOT fail for:
- Reasonable inferences from cited sources
- Missing citations on general industry knowledge
- Imprecise numbers that are directionally correct
- Professional opinions or assessments

This is a pass/warn system — only fail if there is a CRITICAL issue.
Respond with ONLY this JSON (no markdown):
{"passed": true/false, "feedback": "<critical issue only, empty string if passed>"}
""".strip()


def safety_node(state: AgentState) -> Dict:
    """
    Quality gate — flags only critical hallucinations or off-topic answers.
    Safety is a WARNING system, NOT a retry trigger.
    The answer always passes through; safety_feedback is shown as a UI warning.
    """
    draft = state.get("draft_answer", "")
    query = state["query"]

    logger.info("[SAFETY] evaluating answer")

    if not draft.strip():
        logger.warning("[SAFETY] empty draft — passing through with warning")
        return {
            "safety_passed": True,
            "safety_feedback": "Warning: empty answer generated",
            "final_answer": "I was unable to generate an answer. Please try rephrasing your question.",
        }

    from src.agents.moe_router import EXPERT_CONFIGS
    safety_llm = get_llm(EXPERT_CONFIGS["safety_check"])

    try:
        response = safety_llm.invoke(
            [
                SystemMessage(content=_SAFETY_SYSTEM),
                HumanMessage(content=f"Query: {query}\n\nAnswer:\n{draft[:2000]}"),
            ]
        )
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        result = json.loads(raw)
        passed  = bool(result.get("passed", True))
        feedback = result.get("feedback", "")

        logger.info(f"[SAFETY] passed={passed} feedback='{feedback[:80]}'")

    except Exception as e:
        logger.warning(f"[SAFETY] evaluation error: {e} — defaulting to pass")
        passed   = True
        feedback = ""

    # Always deliver the answer — safety_feedback surfaces as a UI warning only
    return {
        "safety_passed":   True,
        "safety_feedback": feedback if not passed else "",
        "final_answer":    draft,
    }
