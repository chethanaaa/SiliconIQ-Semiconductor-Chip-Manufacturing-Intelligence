from src.rag.retrieval import retrieve_and_rerank


class DummyIndex:
    pass


def test_retrieve_and_rerank_uses_vector_search_then_hybrid_rerank(monkeypatch):
    monkeypatch.setattr(
        "src.rag.retrieval.embed_query",
        lambda query, openai_api_key, model: [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(
        "src.rag.retrieval.search_index",
        lambda index, metadata, query_vector, top_k: [
            {
                "chunk_id": "match-1",
                "text": "Advanced packaging and chiplet capacity constraints.",
                "score": 0.71,
                "topic": "advanced_packaging",
                "doc_type": "industry_report",
                "domain_tags": ["advanced packaging", "chiplet"],
                "section_header": "Packaging",
            },
            {
                "chunk_id": "match-2",
                "text": "Macroeconomic conditions influenced capital spending.",
                "score": 0.73,
                "topic": "investment_and_policy",
                "doc_type": "economic_analysis",
                "domain_tags": ["capex"],
                "section_header": "Capex",
            },
        ],
    )

    results = retrieve_and_rerank(
        query="What packaging and chiplet bottlenecks matter most?",
        index=DummyIndex(),
        metadata=[],
        openai_api_key="test-key",
        candidate_k=4,
        top_k=1,
        rerank_strategy="hybrid",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "match-1"
    assert results[0]["rerank_strategy"] == "hybrid"
