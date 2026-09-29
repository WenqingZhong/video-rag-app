import json
from unittest.mock import MagicMock

import httpx
import pytest
import redis
from sqlalchemy import select

from src.config import Settings
from src.models import LLMCallRecord
from src.services.answering import AskService
from src.services.answering.cache import AnswerCache
from src.services.cache import CacheClient, IndexVersion, code_fingerprint, normalize_request
from src.services.indexing import indexer
from src.services.llm import OllamaChat
from src.services.search import SearchHit, SearchResult
from src.services.search.query_parser import ParsedQuery
from src.services.tracing import start_trace
from src.services.understanding import Intent, LLMIntentParser, QueryUnderstanding, Understanding, UnderstandingCache
from src.services.usage import LLMCall, Pricing, UsageRecorder
from src.services.usage.report import usage_report


class FakeRedis:
    """The three commands the caches use, in memory. `down=True` makes every command fail like an outage."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int | None] = {}
        self.down = False

    def _check(self):
        if self.down:
            raise redis.ConnectionError("redis down")

    def get(self, key):
        self._check()
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self._check()
        self.data[key], self.ttl[key] = value, ex

    def incr(self, key):
        self._check()
        self.data[key] = str(int(self.data.get(key, 0)) + 1)
        return int(self.data[key])


@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest.fixture
def cache(fake_redis):
    return CacheClient(fake_redis)


def recorder(database) -> UsageRecorder:
    return UsageRecorder(database, Pricing("ref", 1.0, 5.0), origin="api")


def rows(database) -> list[LLMCallRecord]:
    with database.get_session() as session:
        return list(session.scalars(select(LLMCallRecord).order_by(LLMCallRecord.id)))


# ---- keys ----------------------------------------------------------------------------------------------------
def test_requests_differing_only_in_case_and_spacing_share_a_key():
    assert normalize_request("  A dog\trunning ") == normalize_request("a DOG running") == "a dog running"


def test_fingerprint_changes_with_the_settings_it_covers():
    a = code_fingerprint("services/understanding", settings={"model": "qwen2.5vl:3b"})
    assert a == code_fingerprint("services/understanding", settings={"model": "qwen2.5vl:3b"})
    assert a != code_fingerprint("services/understanding", settings={"model": "llama3.2:1b"})


# ---- understanding cache ----------------------------------------------------------------------------------------
def llm_that_counts(reply: dict | int):
    """An LLM parser whose calls are counted; `reply` is the intent JSON, or an HTTP status to fail with."""
    calls = []

    def handler(request):
        calls.append(1)
        if isinstance(reply, int):
            return httpx.Response(reply)
        return httpx.Response(200, json={"message": {"content": json.dumps(reply)}, "prompt_eval_count": 423, "eval_count": 22})

    llm = LLMIntentParser(OllamaChat("http://ollama", "qwen2.5vl:3b"))
    llm.chat.http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return llm, calls


DOG = {"type": "visual", "phrase": None, "visual": "a dog", "topic": None, "exclude": []}


def test_a_repeated_request_skips_the_model_and_records_the_saving(cache, fake_redis, database):
    llm, calls = llm_that_counts(DOG)
    understanding = QueryUnderstanding(llm, recorder(database), UnderstandingCache(cache, "qwen2.5vl:3b", 3600))

    first = understanding.understand("a dog")
    saved = []
    with start_trace("POST /api/v1/ask", service="api", save=saved.extend):
        second = understanding.understand("A  dog")

    assert len(calls) == 1  # the second request never reached the model
    assert not first.cached and second.cached and second.intent == first.intent and second.source == first.source
    assert second.llm_call is None and second.avoided_call.total_tokens == 445
    assert list(fake_redis.ttl.values()) == [3600]
    assert [(r.outcome, r.prompt_tokens, r.saved_tokens) for r in rows(database)] == [("accepted", 423, 0), ("cache_hit", 0, 445)]
    [step] = [s for s in saved if s.name == "understand"]
    assert step.attributes["cache"] == "hit" and step.attributes["saved_tokens"] == 445


def test_a_failed_model_call_is_not_cached(cache, database):
    llm, calls = llm_that_counts(503)
    understanding = QueryUnderstanding(llm, recorder(database), UnderstandingCache(cache, "qwen2.5vl:3b", 3600))
    understanding.understand("a dog")
    understanding.understand("a dog")
    assert len(calls) == 2  # the rules answered both times; the model is tried again


def test_redis_down_means_no_cache_not_an_error(cache, fake_redis, database):
    fake_redis.down = True
    llm, _ = llm_that_counts(DOG)
    understanding = QueryUnderstanding(llm, None, UnderstandingCache(cache, "qwen2.5vl:3b", 3600))
    assert understanding.understand("a dog").intent.visual == "a dog"


# ---- answer cache -----------------------------------------------------------------------------------------------
def visual_hit():
    doc = {"segment_id": "s1", "video_id": "v1", "kind": "visual", "start_sec": 0.0, "end_sec": 5.28,
           "frame_time_sec": 2.64, "video_title": "Perro", "video_duration_sec": 5.28, "caption": "A dog."}  # fmt: skip
    return SearchHit(source=doc, score=0.03, match_start_sec=0.0, match_end_sec=5.28, match_score=None,
                     matched_text=None, highlight=None, scores={"vector": 0.3})  # fmt: skip


@pytest.fixture
def ask_parts(cache, database):
    understanding = MagicMock()
    understanding.llm = object()  # a model is configured
    understanding.recorder = recorder(database)
    understood = Understanding(Intent(type="visual", visual="a dog"), "llm", 1.2)
    understood.llm_call = LLMCall("understand", "qwen", 423, 22)
    understanding.understand.return_value = understood
    search = MagicMock()
    search.search_intent.return_value = SearchResult(
        parsed=ParsedQuery("a dog", None, "a dog"), strategy="hybrid", hits=[visual_hit()]
    )
    clips = MagicMock()
    clips.get_or_cut.return_value = ("clips/v1/000000000-000005280.mp4", False)
    clips.storage.exists.return_value = True
    settings = Settings(_env_file=None)
    versions = IndexVersion(cache)
    service = AskService(understanding, search, clips, settings, answers=AnswerCache(cache, versions, settings))
    return service, understanding, search, clips, versions


def test_a_repeated_request_is_answered_from_the_cache(ask_parts, database):
    service, understanding, search, _, _ = ask_parts
    first = service.ask("a dog")
    second = service.ask("A dog ")

    assert search.search_intent.call_count == 1 and understanding.understand.call_count == 1
    assert first.cache == {"answer": "miss", "understanding": "miss"}
    assert second.cache == {"answer": "hit", "understanding": "skipped"}
    assert second.answer == first.answer and second.clips[0].key == first.clips[0].key and second.clips[0].cached
    assert second.clips[0].hit == first.clips[0].hit  # the search hit survives the round trip
    assert set(second.timings) == {"cache", "total"}
    assert second.understanding.avoided_call.total_tokens == 445
    assert [(r.outcome, r.saved_tokens) for r in rows(database)] == [("cache_hit", 445)]


def test_an_index_change_invalidates_cached_answers(ask_parts):
    service, _, search, _, versions = ask_parts
    service.ask("a dog")
    versions.bump()  # a video was indexed: the old answer may be wrong now
    assert service.ask("a dog").cache["answer"] == "miss"
    assert search.search_intent.call_count == 2


def test_filters_are_part_of_the_key(ask_parts):
    service, _, search, _, _ = ask_parts
    service.ask("a dog")
    service.ask("a dog", video_id="v2")
    service.ask("a dog", max_clips=2)
    assert search.search_intent.call_count == 3


def test_a_deleted_clip_file_means_answer_afresh(ask_parts):
    service, _, _, clips, _ = ask_parts
    service.ask("a dog")
    clips.storage.exists.return_value = False
    assert service.ask("a dog").cache["answer"] == "miss"


def test_an_answer_made_after_a_model_failure_is_not_cached(ask_parts):
    service, understanding, search, _, _ = ask_parts
    understanding.understand.return_value = Understanding(Intent(type="visual", visual="a dog"), "rules", 0.1)  # no llm_call
    service.ask("a dog")
    service.ask("a dog")
    assert search.search_intent.call_count == 2


def test_redis_down_answers_without_the_cache(ask_parts, fake_redis):
    service, *_ = ask_parts
    fake_redis.down = True
    assert service.ask("a dog").cache["answer"] == "off"


# ---- invalidation at the source ----------------------------------------------------------------------------------
def test_indexing_a_video_bumps_the_index_version(cache, monkeypatch):
    monkeypatch.setattr(indexer, "segment_documents", lambda video, segments: [{"segment_id": "s1"}])
    database = MagicMock()
    database.get_session.return_value.__enter__.return_value = MagicMock()
    opensearch = MagicMock()
    opensearch.index_video.return_value = 1
    versions = IndexVersion(cache)
    indexer.index_video(database, opensearch, "v1", versions=versions)
    indexer.index_video(database, opensearch, "v1", versions=versions)
    assert versions.current() == 2


# ---- report -------------------------------------------------------------------------------------------------------
def test_report_counts_hits_apart_from_consumption(database):
    rec = recorder(database)
    rec.record(LLMCall("understand", "qwen", 420, 30), "accepted")
    rec.record_saved(LLMCall("understand", "qwen", 420, 30))
    rec.record_saved(LLMCall("understand", "qwen", 420, 30))
    with database.get_session() as session:
        report = usage_report(session)
    assert report["by_operation"]["understand"]["calls"] == 1  # hits don't dilute tokens per call
    assert report["by_operation"]["understand"]["tokens_per_call"] == 450
    assert report["cache"]["understand"] == {"hits": 2, "hit_rate": 0.667, "saved_tokens": 900, "saved_cost_usd": 0.00114}
