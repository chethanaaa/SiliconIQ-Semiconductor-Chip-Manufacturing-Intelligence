from src.agents.graph import run_query


class DummyApp:
    def invoke(self, initial_state, config=None):
        state = dict(initial_state)
        state["final_answer"] = "answer"
        state["citations"] = []
        state["expert_type"] = "quick_factual"
        state["tools_to_call"] = ["rag_retrieval"]
        state["plan"] = {"core_question": initial_state["query"]}
        return state


def test_run_query_records_latency(monkeypatch):
    captured = {}

    def fake_record_query_latency(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr("src.agents.graph.record_query_latency", fake_record_query_latency)

    result = run_query(DummyApp(), "What is CoWoS?", thread_id="session-123")

    assert result["final_answer"] == "answer"
    assert captured["query"] == "What is CoWoS?"
    assert captured["thread_id"] == "session-123"
    assert captured["success"] is True
    assert captured["latency_ms"] >= 0
