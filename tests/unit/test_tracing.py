import json
from types import SimpleNamespace

import httpx
import pytest

from src.services.llm import OllamaChat
from src.services.tracing import TraceStore, current_trace_id, get_trace, list_traces, span, start_trace, waterfall
from src.services.understanding import LLMIntentParser, QueryUnderstanding
from src.services.usage import LLMCall, Pricing, UsageRecorder
from src.worker import tracing as worker_tracing


def collect() -> tuple[list, callable]:
    saved: list = []
    return saved, saved.extend


def test_spans_nest_and_are_timed():
    saved, save = collect()
    with start_trace("POST /ask", service="api", save=save) as root:
        with span("understand") as outer, span("llm.understand") as inner:
            inner.set(tokens=448, ignored=None)
        with span("search"):
            pass
    by_name = {s.name: s for s in saved}
    assert [s.name for s in saved] == ["POST /ask", "understand", "llm.understand", "search"]
    assert by_name["understand"].parent_id == root.span_id and by_name["llm.understand"].parent_id == outer.span_id
    assert by_name["search"].parent_id == root.span_id  # back at the top level after the first span closed
    assert by_name["llm.understand"].attributes == {"tokens": 448}  # None values are dropped
    assert all(s.duration_ms is not None and s.trace_id == root.trace_id for s in saved)
    assert current_trace_id() is None  # the context is restored afterwards


def test_an_exception_marks_the_span_and_its_parents():
    saved, save = collect()
    with pytest.raises(ValueError), start_trace("task", service="worker", save=save), span("stage.probing"):
        raise ValueError("corrupt file")
    assert [(s.name, s.status) for s in saved] == [("task", "error"), ("stage.probing", "error")]
    assert saved[1].error == "ValueError: corrupt file"


def test_span_outside_a_trace_does_nothing():
    with span("search") as step:
        step.set(hits=3)  # no error, nothing recorded
    assert current_trace_id() is None


def test_a_failed_save_never_fails_the_request():
    def broken(spans):
        raise RuntimeError("database down")

    with start_trace("GET /x", service="api", save=broken):
        pass  # no exception


# ---- storing and reading ------------------------------------------------------------------------------------
def test_store_round_trip_with_usage_and_waterfall(database):
    store = TraceStore(database)
    recorder = UsageRecorder(database, Pricing("ref", 1.0, 5.0), origin="api")
    with start_trace("POST /api/v1/ask", service="api", trace_id="req-12345678", save=store.save):
        with span("understand"):
            recorder.record(LLMCall("understand", "qwen", 420, 28), "accepted")  # request_id comes from the trace
        with span("search", odd=object()):  # not JSON: stored as its string, the trace isn't lost
            pass

    with database.get_session() as session:
        trace = get_trace(session, "req-12345678")
        recent = list_traces(session, name="POST /api/v1/ask")
    assert [s["name"] for s in trace["spans"]] == ["POST /api/v1/ask", "understand", "search"]
    assert trace["llm"] == {"calls": 1, "tokens": 448, "cost_usd": 0.00056}
    assert [t["trace_id"] for t in recent] == ["req-12345678"]  # roots only
    lines = waterfall(trace).splitlines()
    assert lines[0].startswith("POST /api/v1/ask") and lines[1].startswith("  understand")


# ---- request understanding ------------------------------------------------------------------------------------
def test_understanding_spans_carry_tokens_and_outcome():
    llm = LLMIntentParser(OllamaChat("http://ollama", "qwen2.5vl:3b"))
    body = {"type": "visual", "phrase": None, "visual": "a dog", "topic": None, "exclude": []}
    reply = {"message": {"content": json.dumps(body)}, "prompt_eval_count": 423, "eval_count": 22}
    llm.chat.http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply)), base_url="http://o")
    saved, save = collect()
    with start_trace("POST /api/v1/ask", service="api", save=save):
        QueryUnderstanding(llm).understand("a dog")
    by_name = {s.name: s for s in saved}
    assert by_name["understand"].attributes["outcome"] == "accepted"
    assert by_name["understand"].attributes["tokens"] == 445
    assert by_name["llm.understand"].attributes["prompt_tokens"] == 423


def test_understanding_span_records_an_llm_failure():
    llm = LLMIntentParser(OllamaChat("http://ollama", "qwen2.5vl:3b"))
    llm.chat.http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)), base_url="http://o")
    saved, save = collect()
    with start_trace("POST /api/v1/ask", service="api", save=save):
        result = QueryUnderstanding(llm).understand("a dog")
    assert result.source == "rules"
    assert {s.name: s.status for s in saved} == {"POST /api/v1/ask": "ok", "understand": "ok", "llm.understand": "error"}


# ---- across the queue -------------------------------------------------------------------------------------------
def fake_task(name: str, headers: dict, hostname: str = "clips@abc"):
    request = SimpleNamespace(hostname=hostname, retries=0, **headers)
    return SimpleNamespace(name=name, request=request)


def test_trace_continues_in_the_worker(monkeypatch):
    headers: dict = {}
    with start_trace("POST /api/v1/ask", service="api") as root, span("clip") as clip:
        worker_tracing._add_trace_headers(headers=headers)  # what Celery calls when .delay() publishes
    assert headers == {"trace_id": root.trace_id, "parent_span_id": clip.span_id}

    saved, save = collect()
    monkeypatch.setattr(worker_tracing, "TraceStore", lambda db: SimpleNamespace(save=save))
    monkeypatch.setattr("src.worker.context.get_database", lambda: None)
    task = fake_task("clip.cut", headers)
    worker_tracing._start("t1", task, args=("v1", 1.0, 3.0))
    with span("ffmpeg.cut"):
        pass
    worker_tracing._finish("t1", task, state="SUCCESS")

    worker_root, cut = saved
    assert worker_root.name == "task clip.cut" and worker_root.service == "clip-worker"
    assert worker_root.trace_id == root.trace_id and worker_root.parent_id == clip.span_id  # one trace, two processes
    assert worker_root.attributes["video_id"] == "v1" and cut.parent_id == worker_root.span_id


def test_task_started_outside_a_request_gets_its_own_trace(monkeypatch):
    saved, save = collect()
    monkeypatch.setattr(worker_tracing, "TraceStore", lambda db: SimpleNamespace(save=save))
    monkeypatch.setattr("src.worker.context.get_database", lambda: None)
    task = fake_task("video.process", {}, hostname="worker@abc")
    worker_tracing._start("t2", task, kwargs={"video_id": "v9"})
    worker_tracing._failed("t2", RuntimeError("boom"))
    worker_tracing._finish("t2", task, state="FAILURE")
    [root] = saved
    assert root.parent_id is None and len(root.trace_id) == 32 and root.status == "error"
    assert root.service == "worker" and root.attributes["state"] == "FAILURE"


def test_health_check_tasks_are_not_traced():
    worker_tracing._start("t3", fake_task("system.ping", {}), args=())
    assert "t3" not in worker_tracing._active


def test_retention_deletes_only_old_spans(database):
    from datetime import UTC, datetime, timedelta

    from src.services.tracing import delete_spans_before

    store = TraceStore(database)
    with start_trace("GET /old", service="api", trace_id="old-trace-1", save=store.save) as old:
        old.started_at = datetime.now(UTC) - timedelta(days=30)
    with start_trace("GET /new", service="api", trace_id="new-trace-1", save=store.save):
        pass
    with database.get_session() as session:
        assert delete_spans_before(session, datetime.now(UTC) - timedelta(days=14)) == 1
        assert get_trace(session, "old-trace-1") is None and get_trace(session, "new-trace-1") is not None
