from src.rag.reranker import rerank_results


def test_hybrid_reranking_promotes_keyword_and_metadata_matches():
    candidates = [
        {
            "chunk_id": "c1",
            "text": "Wafer fabrication capacity expanded across leading foundries.",
            "score": 0.82,
            "topic": "chip_manufacturing_process",
            "doc_type": "industry_report",
            "domain_tags": ["wafer", "foundry"],
            "section_header": "Manufacturing Capacity",
            "source_file": "report.pdf",
            "page_num": 4,
        },
        {
            "chunk_id": "c2",
            "text": "EDA tooling demand increased for advanced SoC verification.",
            "score": 0.84,
            "topic": "chip_design",
            "doc_type": "technical_analysis",
            "domain_tags": ["EDA", "SoC"],
            "section_header": "Design Tools",
            "source_file": "report.pdf",
            "page_num": 7,
        },
        {
            "chunk_id": "c3",
            "text": "Geopolitical tensions affected export controls and tariff planning.",
            "score": 0.79,
            "topic": "geopolitics_and_trade",
            "doc_type": "geopolitical_analysis",
            "domain_tags": ["export control", "tariff"],
            "section_header": "Trade Risks",
            "source_file": "brief.pdf",
            "page_num": 3,
        },
    ]

    results = rerank_results(
        query="Which wafer foundry manufacturing risks are affecting semiconductor supply?",
        candidates=candidates,
        top_k=2,
        strategy="hybrid",
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]
    assert "lexical_score" in results[0]
    assert "metadata_score" in results[0]


def test_hybrid_reranking_returns_empty_for_no_candidates():
    assert rerank_results("chiplets", [], strategy="hybrid") == []
