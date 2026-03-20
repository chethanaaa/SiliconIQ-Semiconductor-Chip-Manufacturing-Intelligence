"""
Semiconductor Supply Chain Intelligence — Chat UI
"""

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="Semiconductor Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { background-color: #0f1117; font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d2e 0%, #0f1117 100%);
    border-right: 1px solid #2d2f3e;
}
[data-testid="stChatMessage"] {
    background: #1a1d2e; border-radius: 12px;
    border: 1px solid #2d2f3e; margin-bottom: 8px;
}
[data-testid="stChatInput"] textarea {
    background-color: #1a1d2e !important;
    border: 1px solid #3d4f7c !important;
    color: #e0e0e0 !important; border-radius: 10px !important;
}
[data-testid="stMetric"] {
    background: #1a1d2e; border: 1px solid #2d2f3e;
    border-radius: 8px; padding: 8px 12px;
}
details { background: #1a1d2e !important; border: 1px solid #2d2f3e !important; border-radius: 8px !important; }
.tag {
    display: inline-block; background: #1e3a5f; color: #64b5f6;
    border: 1px solid #2d5a8e; border-radius: 4px;
    padding: 2px 8px; font-size: 11px; margin: 2px;
}
.source-card {
    background: #12151f; border: 1px solid #2d2f3e;
    border-left: 3px solid #3d7ef5; border-radius: 6px;
    padding: 10px 14px; margin: 6px 0; font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers defined FIRST (before any call sites) ────────────────────────────

def _render_meta(meta: dict):
    """Render citations below an assistant message."""
    citations = meta.get("citations", [])
    if not citations:
        return

    _COLORS = ["#64b5f6","#81c784","#ffb74d","#e57373","#ba68c8",
               "#4dd0e1","#f06292","#aed581","#ffcc02","#80cbc4"]

    with st.expander(f"📎 {len(citations)} Source{'s' if len(citations)>1 else ''}", expanded=False):
        for c in citations:
            n      = c.get("citation_number", "?")
            color  = _COLORS[(int(n) - 1) % len(_COLORS)]
            src    = c.get("source_type", "rag")
            excerpt = c.get("excerpt", "")
            tags   = c.get("domain_tags", [])[:4]
            tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags)

            if src == "news":
                pub = c.get("published_at", "")
                st.markdown(f"""
<div class="source-card">
  <strong style="color:{color}">[{n}]</strong> 📰 <strong>{c.get('document','')}</strong>
  &nbsp;<span style="color:#aaa;font-size:12px">{pub}</span><br>
  <span style="font-size:13px">{c.get('title','')}</span><br>
  <span style="color:#888;font-size:12px;font-style:italic">"{excerpt}"</span>
</div>""", unsafe_allow_html=True)
            elif src == "fred":
                st.markdown(f"""
<div class="source-card">
  <strong style="color:{color}">[{n}]</strong> 📊 <strong>{c.get('document','')}</strong><br>
  <span style="font-size:13px">{c.get('title','')}</span>
  &nbsp;<span style="color:#aaa;font-size:12px">· {c.get('units','')} · {c.get('frequency','')}</span><br>
  <span style="color:#888;font-size:12px">Last updated: {c.get('last_updated','')}</span>
</div>""", unsafe_allow_html=True)
            else:
                doc     = c.get("document", "unknown")
                page    = c.get("page", 0)
                total   = c.get("total_pages", 0)
                section = c.get("section") or "—"
                topic   = c.get("topic", "general").replace("_", " ").title()
                st.markdown(f"""
<div class="source-card">
  <strong style="color:{color}">[{n}]</strong> 📄 <strong>{doc}</strong>
  &nbsp;<span style="color:#aaa;font-size:12px">p.{page}/{total} · {section}</span><br>
  <span style="color:#888;font-size:12px">Topic: {topic}</span>&nbsp;{tag_html}<br>
  <span style="color:#777;font-size:12px;font-style:italic">"{excerpt}"</span>
</div>""", unsafe_allow_html=True)


def _clean_answer(text: str) -> str:
    """Escape bare $ signs and strip markdown heading syntax (#, ##, ###)."""
    import re
    # Remove markdown headings (replace with bold instead)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
    # Escape bare $ to avoid LaTeX rendering
    text = re.sub(r'(?<!\$)\$(?!\$)', r'\\$', text)
    return text


def _escape_dollars(text: str) -> str:
    return _clean_answer(text)


def _stream_run_query(app, query: str, thread_id: str, status) -> dict:
    """
    Stream the LangGraph graph node-by-node, updating the status widget
    after each node completes. Returns the FULL accumulated state dict.
    """
    from src.agents.graph import _build_initial_state

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = _build_initial_state(query)

    node_messages = {
        "planner":       ("🗺️", "Plan built — calling data tools..."),
        "tool_executor": ("🔍", "Research complete — routing to expert model..."),
        "moe_router":    ("🧠", "Expert selected — running domain agents..."),
        "synthesizer":   ("✍️", "Answer drafted — running safety check..."),
        "safety":        ("🛡️", "Safety check complete — finalising..."),
    }

    for chunk in app.stream(initial_state, config=config, stream_mode="updates"):
        node_name = next(iter(chunk))
        icon, msg = node_messages.get(node_name, ("⚙️", f"{node_name} complete..."))
        status.update(label=f"Running agentic pipeline... {icon}", expanded=True)
        status.write(f"{icon} {msg}")

    # Return FULL accumulated state (not just the last node's delta)
    return app.get_state(config).values


# ── Cache graph load ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading indexes and building agent graph...")
def load_graph():
    from src.agents.graph import build_graph
    return build_graph(
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "data/vector_store/faiss_index"),
        metadata_path=os.getenv("FAISS_METADATA_PATH", "data/vector_store/metadata.json"),
    )


# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Semiconductor Intelligence")
    st.markdown("*Agentic RAG — Supply Chain & Chip Design*")
    st.divider()

    faiss_path = Path(os.getenv("FAISS_INDEX_PATH", "data/vector_store/faiss_index"))
    bm25_path  = Path("data/vector_store/bm25_index.pkl")
    indexes_ready = faiss_path.exists() and bm25_path.exists()

    if indexes_ready:
        st.success("Indexes loaded", icon="✅")
    else:
        st.error("Indexes not found", icon="❌")
        st.code("python run_ingestion.py", language="bash")

    st.divider()
    st.markdown("**Session**")
    st.markdown(f"`{st.session_state.thread_id}`")
    if st.button("New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**📚 Document Corpus**")
    for pdf in sorted(Path("data/raw").glob("*.pdf")):
        st.caption(f"• {pdf.name}")

    st.divider()
    st.markdown("**🧠 MoE Expert Routing**")
    st.caption("🔵 Deep Reasoning → Claude Sonnet")
    st.caption("🟢 Market Analysis → Claude Sonnet")
    st.caption("🟡 Synthesis → GPT-4o")
    st.caption("🟠 Data Interpretation → GPT-4o")
    st.caption("⚪ Quick Factual → GPT-4o-mini")
    st.divider()
    st.caption("Tools: RAG · NewsAPI · FRED")
    st.caption("Agents: Planner · Procurement · Risk · Manufacturing · Synthesizer · Safety")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ Semiconductor Supply Chain Intelligence")
st.markdown(
    "Ask anything about **semiconductor supply chains**, **chip manufacturing**, "
    "**market dynamics**, **geopolitical risks**, or **procurement strategy**."
)
st.divider()

# ── Load graph ────────────────────────────────────────────────────────────────
if not indexes_ready:
    st.warning("⚠️ Vector indexes not found. Run `python run_ingestion.py` first, then refresh.")
    st.stop()

try:
    graph = load_graph()
except Exception as e:
    st.error(f"Failed to load graph: {e}")
    st.stop()


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚡"):
        st.markdown(_clean_answer(msg["content"]) if msg["role"] == "assistant" else msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            _render_meta(msg["meta"])


# ── Suggested prompts ─────────────────────────────────────────────────────────
EXAMPLES = [
    "What are the biggest supply chain risks for advanced semiconductor manufacturing?",
    "How dependent is the US on TSMC for leading-edge chips?",
    "What is the current state of EUV lithography supply?",
    "How has the CHIPS Act impacted semiconductor manufacturing investment?",
]

if not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        with cols[i % 2]:
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state._pending_query = q
                st.rerun()

if hasattr(st.session_state, "_pending_query"):
    user_input = st.session_state._pending_query
    del st.session_state._pending_query
else:
    user_input = st.chat_input("Ask about semiconductors, supply chain, manufacturing, markets...")


# ── Handle query ──────────────────────────────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="⚡"):
        with st.status("Running agentic pipeline...", expanded=True) as status:
            status.write("🗺️ Planner decomposing query...")
            try:
                final_state = _stream_run_query(graph, user_input, st.session_state.thread_id, status)
                status.update(label="✅ Complete", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Pipeline error: {e}")
                st.stop()

        answer          = final_state.get("final_answer") or final_state.get("draft_answer") or "No answer generated."
        citations       = final_state.get("citations", [])
        safety_passed   = final_state.get("safety_passed", True)
        safety_feedback = final_state.get("safety_feedback", "")
        meta = {
            "expert_type":    final_state.get("expert_type", ""),
            "tools_called":   final_state.get("tools_to_call", []),
            "agents_called":  final_state.get("plan", {}).get("agents_to_call", []),
            "safety_passed":  safety_passed,
            "safety_feedback": safety_feedback,
            "citations":      citations,
        }

        st.markdown(_clean_answer(answer))
        _render_meta(meta)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": meta,
        "safety_feedback": safety_feedback,
    })
