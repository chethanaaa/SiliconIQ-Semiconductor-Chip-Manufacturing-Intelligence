"""
CrewAI Agents & Tasks
---------------------
Five specialist agents for the semiconductor supply chain agentic RAG system.

  PlannerAgent       — decomposes query, selects tools + domain agents
                       LLM: Claude Sonnet (deep reasoning)

  ProcurementAgent   — sourcing, supplier relationships, pricing, lead times
                       LLM: GPT-4o (structured, data-driven)

  RiskAgent          — supply chain risk, geopolitical exposure, disruption
                       LLM: Claude Sonnet (causal + adversarial reasoning)

  ManufacturingAgent — fab processes, technology nodes, yield, packaging
                       LLM: Claude Sonnet (technical depth)

  SynthesizerAgent   — assembles domain agent outputs into a final cited answer
                       LLM: MoE-selected

Crew execution model:
  Planner → [Procurement | Risk | Manufacturing] (only those relevant) → Synthesizer
  All domain agents run sequentially; each receives prior agent output as context.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Planner ───────────────────────────────────────────────────────────────────

_PLANNER_BACKSTORY = """
You are a senior research strategist with 15 years of experience in the
semiconductor and chip design supply chain industry. You have deep expertise in:
- Global semiconductor market dynamics (SIA, SEMI, McKinsey reports)
- Supply chain risk assessment (single-source dependencies, geographic concentration)
- Chip manufacturing processes (fab nodes, EUV/DUV lithography, advanced packaging)
- Geopolitical impacts on trade (CHIPS Act, export controls, China-Taiwan tensions)
- Economic indicators relevant to the industry (FRED data, PPI, production indices)
- Procurement dynamics (supplier qualification, long-term agreements, pricing)

Your role is to analyze incoming research questions, determine which domain experts
to engage, and produce a precise structured research plan.
""".strip()

_PLANNER_GOAL = """
Decompose the user's query into a structured research plan specifying:
1. The primary task type
2. Which data source tools to call
3. Which domain agents are relevant (procurement / risk / manufacturing)
4. Targeted sub-queries for each tool and agent
Output ONLY valid JSON — no prose, no markdown fences.
""".strip()


# ── Procurement Agent ─────────────────────────────────────────────────────────

_PROCUREMENT_BACKSTORY = """
You are a Director of Strategic Sourcing with 18 years of experience in the
semiconductor and electronics procurement industry. Your expertise spans:

SUPPLIER MANAGEMENT
- Qualification and auditing of semiconductor suppliers (IDMs, fabless, OSAT)
- Long-term supply agreements (LTAs), pricing negotiations, volume commitments
- Dual-sourcing and multi-sourcing strategies to reduce single-vendor exposure
- Approved vendor lists (AVL) management across TSMC, Samsung, GlobalFoundries,
  UMC, SMIC, Intel Foundry, and tier-2 specialty fabs

PROCUREMENT OPERATIONS
- Lead time management (standard 12–52 week horizons for advanced nodes)
- Spot market dynamics and grey-market risk during shortage cycles
- Bill of Materials (BOM) cost optimization and should-cost modelling
- Strategic buffer stock and safety stock policies
- Die bank and wafer bank programs for demand smoothing

MARKET INTELLIGENCE
- Semiconductor commodity price trends (DRAM, NAND, logic, analog, power)
- Foundry capacity allocation and priority access programs
- Impact of export controls and trade restrictions on sourcing strategy
- Alternative component qualification timelines and costs

You provide precise, actionable procurement intelligence grounded in industry data.
""".strip()

_PROCUREMENT_GOAL = """
Analyze the query from a procurement and sourcing lens:
- Identify key supplier dependencies and concentration risks
- Assess lead times, pricing dynamics, and capacity availability
- Recommend sourcing strategies (dual-source, LTA, buffer stock)
- Flag procurement risks from geopolitical or regulatory changes
Ground all analysis in the provided context documents.
""".strip()


# ── Risk Agent ────────────────────────────────────────────────────────────────

_RISK_BACKSTORY = """
You are a Chief Supply Chain Risk Officer with 20 years of experience identifying,
quantifying, and mitigating risks across the global semiconductor supply chain.

RISK DOMAINS YOU COVER

Geopolitical & Trade Risk
- Export control regimes (BIS Entity List, EAR, ITAR, Wassenaar Arrangement)
- US–China technology decoupling and its cascading supply chain effects
- Taiwan Strait scenarios and TSMC concentration risk
- CHIPS Act subsidies reshaping fab geography (Intel Arizona, TSMC Arizona/Japan)
- Sanctions exposure across tier-1 and tier-2 suppliers

Supply Concentration Risk
- Single-source dependencies (ASML EUV, JSSI photoresist, Shin-Etsu silicon)
- Geographic concentration: Taiwan ~90% of leading-edge logic
- Choke-point materials: neon gas (Ukraine), palladium (Russia), rare earths (China)

Operational & Demand Risk
- Semiconductor cycle volatility (boom-bust inventory cycles)
- Demand-supply mismatch in automotive, AI, and consumer segments
- Natural disaster exposure (fab concentration in earthquake/typhoon zones)
- Cybersecurity risks in chip design IP and supply chain data

Financial Risk
- Customer credit risk and demand visibility
- FX exposure in multi-currency supply chains
- Capex cycle risk for fab investments ($20B+ commitments)

You quantify risks with likelihood and impact scores where data permits,
and always propose mitigating actions.
""".strip()

_RISK_GOAL = """
Analyze the query through a comprehensive risk lens:
- Identify and rank supply chain risks (geopolitical, operational, financial, regulatory)
- Assess likelihood and potential impact of each risk
- Map choke points and single-source dependencies
- Propose concrete mitigation strategies
- Flag early warning signals from news or economic data
Ground all risk assessments in the provided context and data.
""".strip()


# ── Manufacturing / Process Agent ─────────────────────────────────────────────

_MANUFACTURING_BACKSTORY = """
You are a VP of Process Engineering with 22 years of experience in semiconductor
fabrication and advanced chip manufacturing across leading-edge and mature nodes.

PROCESS TECHNOLOGY EXPERTISE

Front-End-of-Line (FEOL) Manufacturing
- Logic process nodes: from 28nm planar down to 2nm/1.4nm GAA (Gate-All-Around)
- Memory process: DRAM (1γ/1δ node), NAND (200+ layer 3D stacking)
- Lithography: DUV (193nm immersion), EUV (13.5nm), High-NA EUV
- Key process steps: CVD, PVD, ALD, CMP, wet/dry etch, ion implantation
- Yield management: defect density, parametric yield, OPC/RET optimization
- Equipment suppliers: ASML, Applied Materials, Lam Research, KLA, Tokyo Electron

Back-End-of-Line (BEOL) & Packaging
- Advanced packaging: CoWoS, InFO-PoP, FOPLP, SoIC (3D stacking)
- Chiplet architectures: UCIe standard, AMD MI300X, Intel Meteor Lake
- OSAT landscape: ASE, Amkor, JCET, Powertech
- Substrate and interposer supply constraints
- HBM memory integration (HBM3, HBM3E) for AI accelerators

MANUFACTURING OPERATIONS
- Fab capacity planning: wafer starts, WIP management, cycle time
- Yield ramp curves and learning rate models
- Capex intensity and depreciation economics of leading-edge fabs
- Equipment lead times (ASML EUV: 18–24 months)
- Technology transfer and qualification between fabs

You provide precise technical analysis grounded in manufacturing realities,
not marketing claims.
""".strip()

_MANUFACTURING_GOAL = """
Analyze the query from a manufacturing and process technology lens:
- Assess technology readiness and maturity of relevant process nodes
- Identify manufacturing bottlenecks (yield, equipment, materials, capacity)
- Evaluate packaging and integration options (chiplet, advanced packaging)
- Quantify capex requirements and ramp timelines
- Highlight technology differentiators between leading foundries
Ground all analysis in the provided context documents and data.
""".strip()


# ── Synthesizer Agent ─────────────────────────────────────────────────────────

_SYNTHESIZER_BACKSTORY = """
You are an Executive Research Director at a top-tier semiconductor industry
advisory firm. You specialize in synthesizing complex, multi-disciplinary analysis
from procurement, risk, and manufacturing experts into clear, decision-ready reports.

Your synthesis style:
- Lead with a crisp executive summary (3–4 sentences capturing the key answer)
- Integrate insights from all domain experts without repetition
- Highlight where expert analyses converge (high confidence) and diverge (uncertainty)
- Present quantitative data with appropriate precision and caveats
- Structure findings for a CXO or senior analyst audience
- Cite every factual claim with [N] notation referencing the numbered sources
- Close with strategic implications or recommended next steps

You never hallucinate data. If the sources are insufficient, you explicitly state
what is known, what is uncertain, and what additional data would be needed.
""".strip()

_SYNTHESIZER_GOAL = """
Synthesize all domain agent analyses and source context into one comprehensive,
well-structured, and fully cited answer. Lead with an executive summary,
follow with integrated analysis, and close with strategic implications.
Reference every factual claim with [N] inline citations.
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
# CREW RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_fences(raw: str) -> str:
    """Remove markdown code fences from LLM output."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _use_lightweight_agent_execution() -> bool:
    """
    Use direct LLM calls instead of CrewAI orchestration on the latency-sensitive
    query path. This keeps the prompts/roles but avoids repeated Crew startup cost.
    """
    return os.getenv("LIGHTWEIGHT_AGENT_EXECUTION", "true").strip().lower() != "false"


def _response_to_text(response: Any) -> str:
    """Normalize LangChain model responses into plain text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                text = getattr(item, "text", "")
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _invoke_llm_text(
    llm,
    system_prompt: str,
    user_prompt: str,
    label: str,
) -> str:
    """Call a LangChain chat model directly and return plain text."""
    t0 = time.perf_counter()
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    text = _response_to_text(response)
    elapsed = time.perf_counter() - t0
    logger.info(f"[{label}] lightweight LLM call completed in {elapsed:.2f}s")
    return text


# ── Planner ───────────────────────────────────────────────────────────────────

def run_planner(query: str, llm) -> Dict[str, Any]:
    """
    Run the Planner agent to decompose the query into a structured research plan.

    Returns a plan dict with schema:
    {
      "task_type"        : str,
      "tools_to_call"    : [str, ...],
      "agents_to_call"   : [str, ...],   # subset of: procurement, risk, manufacturing
      "rag_query"        : str | null,
      "news_query"       : str | null,
      "fred_series_key"  : str | null,
      "core_question"    : str,
      "reasoning"        : str
    }
    """
    prompt = f"""
Analyze this user query and produce a research plan as JSON.

Query: {query}

Available data source tools:
- rag_retrieval    : searches internal PDF corpus (SIA, McKinsey, supply chain reports)
- fetch_news       : fetches recent semiconductor news via NewsAPI
- fetch_fred_data  : fetches FRED economic time-series data

Available domain agents (select those relevant to the query):
- procurement      : supplier sourcing, pricing, lead times, dual-sourcing strategy
- risk             : geopolitical risk, supply concentration, disruption analysis
- manufacturing    : fab processes, technology nodes, yield, advanced packaging

Valid task_type values:
  deep_reasoning | market_analysis | data_interpretation | synthesis_narration | quick_factual

Return ONLY this JSON (no markdown fences):
{{
  "task_type"       : "<task_type>",
  "tools_to_call"   : ["<tool1>"],
  "agents_to_call"  : ["procurement", "risk"],
  "rag_query"       : "<specific retrieval query or null>",
  "news_query"      : "<specific news query or null>",
  "fred_series_key" : "<series key or null>",
  "core_question"   : "<the core analytical question to answer>",
  "reasoning"       : "<brief explanation of agent and tool selection>"
}}
    """.strip()

    if _use_lightweight_agent_execution():
        result = _invoke_llm_text(
            llm=llm,
            system_prompt=f"{_PLANNER_BACKSTORY}\n\n{_PLANNER_GOAL}",
            user_prompt=prompt,
            label="PLANNER",
        )
    else:
        agent = Agent(
            role="Semiconductor Supply Chain Research Planner",
            goal=_PLANNER_GOAL,
            backstory=_PLANNER_BACKSTORY,
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        task = Task(
            description=prompt,
            agent=agent,
            expected_output="A JSON object with the research plan",
        )

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()

    try:
        plan = json.loads(_strip_fences(str(result)))
        logger.success(
            f"Planner: task_type={plan.get('task_type')} "
            f"tools={plan.get('tools_to_call')} "
            f"agents={plan.get('agents_to_call')}"
        )
        return plan
    except json.JSONDecodeError:
        logger.warning("Planner output not valid JSON — using fallback plan")
        return {
            "task_type": "synthesis_narration",
            "tools_to_call": ["rag_retrieval"],
            "agents_to_call": ["risk", "procurement"],
            "rag_query": query,
            "news_query": None,
            "fred_series_key": None,
            "core_question": query,
            "reasoning": "Fallback: planner output parse failed",
        }


# ── Domain Agent Crew ─────────────────────────────────────────────────────────

def run_domain_agents(
    query: str,
    core_question: str,
    context_block: str,
    agents_to_call: List[str],
    procurement_llm,
    risk_llm,
    manufacturing_llm,
) -> Dict[str, str]:
    """
    Run the selected domain agents as a sequential CrewAI crew.
    Each agent receives the shared context block and the previous agent's output.

    Args:
        query             : original user query
        core_question     : refined question from planner
        context_block     : numbered source chunks + news + FRED data
        agents_to_call    : subset of ["procurement", "risk", "manufacturing"]
        procurement_llm   : LangChain LLM for procurement agent
        risk_llm          : LangChain LLM for risk agent
        manufacturing_llm : LangChain LLM for manufacturing agent

    Returns:
        Dict mapping agent name → analysis string
    """
    agents_to_call = [a for a in agents_to_call
                      if a in ("procurement", "risk", "manufacturing")]

    if not agents_to_call:
        logger.info("No domain agents selected — skipping domain crew")
        return {}

    logger.info(f"[DOMAIN CREW] running agents: {agents_to_call}")

    active_agents: List[Agent] = []
    active_tasks:  List[Task]  = []

    # Trim context to avoid slow/expensive LLM calls — 2000 chars is sufficient
    trimmed_context = context_block[:2000] + ("..." if len(context_block) > 2000 else "")

    shared_context_header = f"""
User Query    : {query}
Core Question : {core_question}

─── SOURCE CONTEXT (excerpt) ─────────────────────────────────────
{trimmed_context}
──────────────────────────────────────────────────────────────────
    """.strip()

    prompts: Dict[str, str] = {}

    # ── Procurement Agent ─────────────────────────────────────────────────
    if "procurement" in agents_to_call:
        prompts["procurement"] = f"""
{shared_context_header}

Your role: Procurement & Sourcing Analysis

Analyze the query from a procurement perspective. Address in 3-5 concise bullet points:
1. Key suppliers relevant to this query
2. Lead times and capacity constraints
3. Sourcing strategy recommendation
4. Top procurement risk

Be concise — maximum 300 words. Reference sources with [N] notation.
        """.strip()
        procurement_agent = Agent(
            role="Semiconductor Procurement & Sourcing Director",
            goal=_PROCUREMENT_GOAL,
            backstory=_PROCUREMENT_BACKSTORY,
            llm=procurement_llm,
            verbose=True,
            allow_delegation=False,
        )
        procurement_task = Task(
            description=prompts["procurement"],
            agent=procurement_agent,
            expected_output="Procurement analysis with sourcing insights and recommendations, cited with [N]",
        )
        active_agents.append(procurement_agent)
        active_tasks.append(procurement_task)

    # ── Risk Agent ────────────────────────────────────────────────────────
    if "risk" in agents_to_call:
        prompts["risk"] = f"""
{shared_context_header}

Your role: Supply Chain Risk Analysis

Analyze the query from a risk perspective. Address in 3-5 concise bullet points:
1. Top geopolitical or concentration risk
2. Most critical operational risk
3. Key mitigation recommendation

Be concise — maximum 300 words. Reference sources with [N] notation.
            """.strip()
        risk_agent = Agent(
            role="Semiconductor Supply Chain Risk Officer",
            goal=_RISK_GOAL,
            backstory=_RISK_BACKSTORY,
            llm=risk_llm,
            verbose=True,
            allow_delegation=False,
        )
        risk_task = Task(
            description=prompts["risk"],
            agent=risk_agent,
            expected_output="Risk analysis with ranked risks, impact assessment, and mitigation strategies, cited with [N]",
        )
        active_agents.append(risk_agent)
        active_tasks.append(risk_task)

    # ── Manufacturing Agent ───────────────────────────────────────────────
    if "manufacturing" in agents_to_call:
        prompts["manufacturing"] = f"""
{shared_context_header}

Your role: Manufacturing & Process Technology Analysis

Analyze the query from a manufacturing perspective. Address in 3-5 concise bullet points:
1. Relevant process node or technology status
2. Key manufacturing bottleneck
3. Packaging or integration consideration if relevant

Be concise — maximum 300 words. Reference sources with [N] notation.
            """.strip()
        manufacturing_agent = Agent(
            role="Semiconductor Process & Manufacturing Engineer",
            goal=_MANUFACTURING_GOAL,
            backstory=_MANUFACTURING_BACKSTORY,
            llm=manufacturing_llm,
            verbose=True,
            allow_delegation=False,
        )
        manufacturing_task = Task(
            description=prompts["manufacturing"],
            agent=manufacturing_agent,
            expected_output="Manufacturing analysis with process details, bottlenecks, and technical recommendations, cited with [N]",
        )
        active_agents.append(manufacturing_agent)
        active_tasks.append(manufacturing_task)

    if _use_lightweight_agent_execution():
        outputs: Dict[str, str] = {}
        llms = {
            "procurement": procurement_llm,
            "risk": risk_llm,
            "manufacturing": manufacturing_llm,
        }
        system_prompts = {
            "procurement": f"{_PROCUREMENT_BACKSTORY}\n\n{_PROCUREMENT_GOAL}",
            "risk": f"{_RISK_BACKSTORY}\n\n{_RISK_GOAL}",
            "manufacturing": f"{_MANUFACTURING_BACKSTORY}\n\n{_MANUFACTURING_GOAL}",
        }

        for agent_name in agents_to_call:
            raw = _invoke_llm_text(
                llm=llms[agent_name],
                system_prompt=system_prompts[agent_name],
                user_prompt=prompts[agent_name],
                label=f"{agent_name.upper()} AGENT",
            )
            outputs[agent_name] = raw
            logger.success(f"[{agent_name.upper()} AGENT] {len(raw)} chars generated")

        return outputs

    # ── Run sequential crew ───────────────────────────────────────────────
    crew = Crew(
        agents=active_agents,
        tasks=active_tasks,
        process=Process.sequential,
        verbose=True,
    )

    crew.kickoff()

    # Collect each task's output
    outputs: Dict[str, str] = {}
    agent_names = [a for a in agents_to_call
                   if a in ("procurement", "risk", "manufacturing")]
    for name, task in zip(agent_names, active_tasks):
        raw = str(task.output).strip() if task.output else ""
        outputs[name] = raw
        logger.success(f"[{name.upper()} AGENT] {len(raw)} chars generated")

    return outputs


# ── Synthesizer ───────────────────────────────────────────────────────────────

def run_synthesizer(
    query: str,
    core_question: str,
    context_block: str,
    domain_outputs: Dict[str, str],
    task_type: str,
    llm,
) -> str:
    """
    Run the Synthesizer agent to assemble a final cited answer from:
      - Source context (RAG + news + FRED)
      - Domain agent outputs (procurement / risk / manufacturing)

    Args:
        query          : original user query
        core_question  : refined question from planner
        context_block  : numbered source chunks
        domain_outputs : {agent_name: analysis_text} from run_domain_agents()
        task_type      : influences tone/depth
        llm            : MoE-selected LangChain LLM

    Returns:
        Final answer string with inline [N] citations.
    """
    tone_guide = {
        "deep_reasoning":      "Deep analytical, causal reasoning, identify risks and implications",
        "market_analysis":     "Data-driven, quantitative, with trends and forward-looking insights",
        "data_interpretation": "Interpret data clearly, contextualise within industry dynamics",
        "synthesis_narration": "Clear, structured narrative — executive summary then analysis",
        "quick_factual":       "Concise and direct — bullets preferred over long paragraphs",
    }.get(task_type, "Clear, structured, and well-cited")

    # Format domain agent outputs as additional context
    domain_section = ""
    if domain_outputs:
        parts = []
        for agent_name, analysis in domain_outputs.items():
            if analysis:
                parts.append(
                    f"── {agent_name.upper()} AGENT ANALYSIS ──────────────────────\n{analysis}"
                )
        if parts:
            domain_section = "\n\n".join(parts)

    agent = Agent(
        role="Semiconductor Industry Research Director",
        goal=_SYNTHESIZER_GOAL,
        backstory=_SYNTHESIZER_BACKSTORY,
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
User query    : {query}
Core question : {core_question}
Tone / style  : {tone_guide}

─── SOURCE DOCUMENTS ─────────────────────────────────────────────
{context_block}
──────────────────────────────────────────────────────────────────

{f'─── DOMAIN EXPERT ANALYSES ──────────────────────────────────────{chr(10)}{domain_section}{chr(10)}──────────────────────────────────────────────────────────────────' if domain_section else ''}

Synthesis instructions:
1. Answer directly in 150–250 words. No lengthy preambles.
2. Use short paragraphs or bullets — no walls of text.
3. Reference factual claims with [N] notation matching the numbered sources above.
4. Integrate domain agent insights briefly where relevant.
5. Do NOT invent facts or statistics absent from the sources.
6. End with 1–2 concise strategic takeaways (1 sentence each).
        """.strip(),
        agent=agent,
        expected_output="Concise answer (150-250 words) with [N] inline citations and 1-2 strategic takeaways",
    )

    if _use_lightweight_agent_execution():
        answer = _invoke_llm_text(
            llm=llm,
            system_prompt=f"{_SYNTHESIZER_BACKSTORY}\n\n{_SYNTHESIZER_GOAL}",
            user_prompt=task.description,
            label="SYNTHESIZER",
        )
        logger.success(f"Synthesizer: {len(answer)} char answer generated")
        return answer

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = crew.kickoff()
    answer = str(result).strip()
    logger.success(f"Synthesizer: {len(answer)} char answer generated")
    return answer
