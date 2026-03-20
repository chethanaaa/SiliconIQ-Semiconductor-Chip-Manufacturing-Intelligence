from datetime import datetime, timezone

from src.evaluation.langsmith_monitor import record_query_latency


def test_record_query_latency_submits_run_when_enabled(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            captured["client_kwargs"] = kwargs

        def create_run(self, **kwargs):
            captured["run"] = kwargs

    monkeypatch.setenv("LANGSMITH_MONITORING_ENABLED", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")
    monkeypatch.setattr("src.evaluation.langsmith_monitor.Client", DummyClient)

    ok = record_query_latency(
        query="What are the main wafer bottlenecks?",
        thread_id="thread-1",
        latency_ms=123.45,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        success=True,
    )

    assert ok is True
    assert captured["run"]["name"] == "user_query_latency"
    assert captured["run"]["project_name"] == "test-project"
    assert captured["run"]["outputs"]["latency_ms"] == 123.45


def test_record_query_latency_skips_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_MONITORING_ENABLED", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert record_query_latency(
        query="test",
        thread_id="thread-1",
        latency_ms=10.0,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        success=True,
    ) is False
