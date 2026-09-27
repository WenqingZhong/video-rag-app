import json
from unittest.mock import MagicMock

import httpx
import pytest
from sqlalchemy import select

from src.models import LLMCallRecord
from src.services.processing.visual import VisualEnricher
from src.services.understanding import LLMIntentParser, QueryUnderstanding
from src.services.usage import LLMCall, Pricing, UsageRecorder

# A real Ollama reply's accounting fields (from a /api/chat call; durations in nanoseconds).
OLLAMA_COUNTS = {
    "prompt_eval_count": 423,
    "eval_count": 26,
    "total_duration": 2_399_326_668,
    "load_duration": 28_736_875,
    "prompt_eval_duration": 68_874_917,
    "eval_duration": 2_250_270_584,
}
PRICING = Pricing("reference", input_per_mtok=1.0, output_per_mtok=5.0)


def test_call_is_read_from_ollama_counts():
    call = LLMCall.from_ollama(OLLAMA_COUNTS, "understand", "qwen")
    assert (call.prompt_tokens, call.output_tokens, call.total_tokens) == (423, 26, 449)
    assert call.prompt_sec == 0.0689 and call.total_sec == 2.3993


def test_missing_counts_are_zero_not_an_error():
    call = LLMCall.from_ollama({}, "caption", "qwen", images=1)
    assert call.total_tokens == 0 and call.images == 1


def test_cost_prices_input_and_output_separately():
    call = LLMCall("understand", "qwen", prompt_tokens=1_000_000, output_tokens=200_000)
    assert PRICING.cost(call) == 1.0 + 1.0  # 1M in × $1 + 0.2M out × $5


def rows(database) -> list[LLMCallRecord]:
    with database.get_session() as session:
        return list(session.scalars(select(LLMCallRecord).order_by(LLMCallRecord.id)))


def test_recorder_stores_tokens_cost_and_context(database):
    call = LLMCall.from_ollama(OLLAMA_COUNTS, "understand", "qwen")
    cost = UsageRecorder(database, PRICING, origin="api").record(call, "accepted", request_id="r1")
    [row] = rows(database)
    assert (row.operation, row.outcome, row.origin, row.request_id) == ("understand", "accepted", "api", "r1")
    assert row.prompt_tokens == 423 and float(row.cost_usd) == cost == pytest.approx(423e-6 + 26 * 5e-6)
    assert row.price_reference == "reference"


def test_recorder_outage_loses_the_row_not_the_request():
    broken = MagicMock()
    broken.get_session.side_effect = RuntimeError("Database not initialized")
    call = LLMCall("caption", "qwen", 199, 35)
    assert UsageRecorder(broken, PRICING, origin="worker").record(call, "ok") > 0  # no exception


def understanding(reply, database) -> QueryUnderstanding:
    """QueryUnderstanding whose LLM returns `reply` (a dict or raw text) with real-looking token counts."""
    llm = LLMIntentParser("http://ollama", model="qwen2.5vl:3b")

    def handler(request):
        content = json.dumps(reply) if isinstance(reply, dict) else reply
        return httpx.Response(200, json={"message": {"content": content}, **OLLAMA_COUNTS})

    llm.http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return QueryUnderstanding(llm, UsageRecorder(database, PRICING, origin="api"))


def reply(type_, phrase=None, visual=None, topic=None, exclude=()):
    return {"type": type_, "phrase": phrase, "visual": visual, "topic": topic, "exclude": list(exclude)}


@pytest.mark.parametrize(
    ("query", "answer", "outcome"),
    [
        ("a dog", reply("visual", visual="a dog"), "accepted"),
        # the rules add the exclusion the model dropped: the answer is used, but changed
        ("a beach with no people", reply("visual", visual="a beach"), "adjusted"),
        # invented details: the answer is thrown away and the rules answer instead
        ("a dog", reply("visual", visual="a cat sleeping on a sofa"), "rejected"),
        ("a dog", "not json", "invalid"),
    ],
)
def test_understanding_records_what_became_of_the_answer(database, query, answer, outcome):
    result = understanding(answer, database).understand(query, request_id="r1")
    assert result.llm_outcome == outcome and result.llm_call.total_tokens == 449 and result.cost_usd > 0
    [row] = rows(database)
    assert (row.outcome, row.request_id) == (outcome, "r1")


def test_rules_only_spends_no_tokens(database):
    result = QueryUnderstanding(None, UsageRecorder(database, PRICING, origin="api")).understand("a dog")
    assert result.llm_call is None and result.cost_usd == 0 and rows(database) == []


def test_enricher_records_one_row_per_caption_with_the_video(database):
    captioner = MagicMock(model="qwen")
    captioner.caption_with_usage.side_effect = [
        ("a dog", LLMCall("caption", "qwen", 199, 35, images=1)),
        ("a cat", LLMCall("caption", "qwen", 199, 36, images=1)),
    ]
    recorder = UsageRecorder(database, PRICING, origin="worker")
    VisualEnricher(None, captioner, recorder).enrich([b"1", b"2"], video_id="v1")
    recorded = rows(database)
    assert [(r.operation, r.outcome, r.video_id, r.images) for r in recorded] == [("caption", "ok", "v1", 1)] * 2
    assert sum(r.output_tokens for r in recorded) == 71


def test_report_answers_the_token_questions(database):
    from src.models import Video
    from src.services.usage.report import usage_report

    with database.get_session() as session:
        session.add(Video(id="v1", source="pexels", title="dogs", duration_sec=30.0, tags=[]))
        session.commit()
    recorder = UsageRecorder(database, PRICING, origin="api")
    recorder.record(LLMCall("understand", "qwen", 420, 30), "accepted")
    recorder.record(LLMCall("understand", "qwen", 420, 30), "rejected")
    for _ in range(3):
        recorder.record(LLMCall("caption", "qwen", 200, 50, images=1), "ok", video_id="v1")

    with database.get_session() as session:
        report = usage_report(session)
    assert report["overall"]["total_tokens"] == 2 * 450 + 3 * 250
    assert report["by_operation"]["understand"]["tokens_per_call"] == 450
    assert report["understand_wasted_share"] == 0.5  # one of two answers thrown away
    [video] = report["videos"]
    assert (video["keyframes"], video["total_tokens"], video["tokens_per_minute"]) == (3, 750, 1500)
    with database.get_session() as session:
        assert usage_report(session, origin="eval")["overall"]["calls"] == 0
