# ⚡ SiliconIQ — Semiconductor Supply Chain Intelligence

> **Ask anything. Know everything.** AI-powered research for semiconductor leaders — instant, cited answers across supply chain risk, procurement strategy, manufacturing intelligence, and market dynamics.

![SiliconIQ Platform](assets/preview.png)

---

## What is SiliconIQ?

SiliconIQ is an **agentic RAG system** purpose-built for the semiconductor and chip design supply chain industry. It fuses deep document research (industry PDFs), live news, and economic data — then routes each query through specialist AI agents to deliver concise, source-cited answers in under 30 seconds.

---

## How It Works

```
You Ask
  → Planner Agent     (Claude Sonnet) — decomposes query, selects tools & agents
  → Tool Execution    — RAG retrieval · Live News · FRED Economic Data
  → Domain Agents     — Procurement · Risk · Manufacturing (run in parallel)
  → Synthesizer       (GPT-4.1-nano) — assembles cited answer
  → Safety Check      (Claude Haiku) — validates before delivery
  → Cited Answer      — every claim traced to page, section, document
```

---

## Knowledge Base

| Source | Description |
|--------|-------------|
| 📄 Industry Reports | SIA State of Industry, McKinsey Semiconductors 2024, Supply Chain Issue Brief — 207 pages, 414 knowledge chunks |
| 📰 Live News Feed | Real-time semiconductor news via NewsAPI — fab investments, export controls, company announcements |
| 📊 FRED Economic Data | Federal Reserve time-series — semiconductor production indices, PPI, capacity utilisation, GDP |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (StateGraph + MemorySaver) |
| Agents | CrewAI (Planner, Procurement, Risk, Manufacturing, Synthesizer) |
| LLM Routing | Mixture of Experts — Claude Sonnet · GPT-4.1-nano · Claude Haiku |
| Retrieval | FAISS (dense) + BM25 (sparse) → Reciprocal Rank Fusion |
| Chunking | LlamaIndex SemanticSplitterNodeParser |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Monitoring | LangSmith tracing |
| UI | Streamlit |

---

## Project Structure

```
cc_template/
├── app/
│   └── main.py                  # Streamlit chat UI
├── src/
│   ├── ingestion/
│   │   ├── pdf_extractor.py     # PyMuPDF + pdfplumber extraction
│   │   ├── chunker.py           # LlamaIndex semantic chunking
│   │   ├── tokenizer.py         # tiktoken cl100k_base
│   │   ├── metadata_enricher.py # Section, topic, domain tag detection
│   │   └── embedder.py          # OpenAI batch embeddings
│   ├── rag/
│   │   ├── vector_store.py      # FAISS IndexFlatIP
│   │   ├── retrieval.py         # Hybrid dense+sparse RRF retrieval
│   │   ├── reranker.py          # Hybrid reranking (vector+lexical+metadata)
│   │   └── citation.py          # Citation dataclass + context builder
│   ├── agents/
│   │   ├── state.py             # LangGraph AgentState
│   │   ├── graph.py             # Graph assembly + build_graph()
│   │   ├── nodes.py             # Planner, ToolExecutor, MoE, Synthesizer, Safety
│   │   ├── crew_agents.py       # CrewAI agent definitions
│   │   ├── moe_router.py        # Expert type classification + LLM routing
│   │   └── tools.py             # RAG, NewsAPI, FRED tool definitions
│   └── evaluation/
│       └── langsmith_monitor.py # Query latency tracing
├── data/
│   ├── raw/                     # Source PDFs
│   ├── processed/               # Chunked + enriched data
│   └── vector_store/            # FAISS index + BM25 index + metadata
├── assets/
│   └── preview.png              # Platform screenshot
├── deck.html                    # Single-slide stakeholder presentation
├── run_ingestion.py             # Full ingestion pipeline runner
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Fill in: OPENAI_API_KEY, ANTHROPIC_API_KEY, NEWS_API_KEY, FRED_API_KEY

# 3. Build vector indexes (run once)
python run_ingestion.py

# 4. Launch
streamlit run app/main.py
```

---

## Environment Variables

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
NEWS_API_KEY=
FRED_API_KEY=
LANGSMITH_API_KEY=          # optional — enables tracing

FAISS_INDEX_PATH=data/vector_store/faiss_index
FAISS_METADATA_PATH=data/vector_store/metadata.json
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
TOP_K_RETRIEVAL=20
TOP_K_RERANK=5
LIGHTWEIGHT_AGENT_EXECUTION=true
```

---

*Built with LangGraph · CrewAI · FAISS · LlamaIndex · Streamlit*
