from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.services.agent.replies import upload_failed
from src.services.cache import CacheClient
from src.services.limits import Limiter, LimitExceeded, Principal, acting_for
from src.services.usage import LLMCall, UsageRecorder
from tests.unit.test_caching import FakeRedis

SETTINGS = Settings(
    _env_file=None,
    limit_tokens_per_day=1000,
    limit_ip_tokens_per_day=3000,
    limit_global_tokens_per_day=10_000,
    limit_requests_per_min=3,
    limit_ip_requests_per_min=5,
    limit_uploads_per_day=2,
    limit_pexels_per_day=1,
)


@pytest.fixture
def redis_():
    return FakeRedis()


@pytest.fixture
def limiter(redis_):
    return Limiter(CacheClient(redis_), SETTINGS)


def test_requests_per_minute_per_viewer(limiter):
    alice = Principal(viewer="web:a", ip="1.1.1.1")
    for _ in range(3):
        limiter.check_request(alice)
    with pytest.raises(LimitExceeded, match="a bit fast"):
        limiter.check_request(alice)


def test_new_anonymous_sessions_from_one_address_share_its_rate(limiter):
    for i in range(5):
        limiter.check_request(Principal(viewer=f"web:{i}", ip="1.1.1.1"))
    with pytest.raises(LimitExceeded):
        limiter.check_request(Principal(viewer="web:new", ip="1.1.1.1"))


def test_the_days_tokens_run_out_after_the_allowance(limiter):
    alice = Principal(viewer="web:a", ip="1.1.1.1")
    limiter.add_tokens(alice, 999)
    limiter.check_request(alice)  # still under: the request that crosses the line finishes
    limiter.add_tokens(alice, 5)
    with pytest.raises(LimitExceeded, match="allowance of 1,000 tokens") as exc:
        limiter.check_request(alice)
    assert 0 < exc.value.retry_after <= 24 * 3600
    limiter.check_request(Principal(viewer="web:b", ip="2.2.2.2"))  # others are unaffected
    assert limiter.usage("web:a").tokens_left == 0


def test_the_global_budget_stops_everyone(limiter):
    limiter.add_tokens(None, 10_000)  # e.g. processing library videos
    with pytest.raises(LimitExceeded, match="overall usage limit"):
        limiter.check_request(Principal(viewer="web:new", ip="3.3.3.3"))


def test_admins_are_exempt(limiter):
    admin = Principal(viewer="web:a", exempt=True)
    limiter.add_tokens(admin, 50_000)
    for _ in range(10):
        limiter.check_request(admin)
        limiter.use_upload(admin)


def test_uploads_and_pexels_downloads_per_day(limiter):
    alice = Principal(viewer="web:a")
    limiter.use_upload(alice)
    limiter.use_upload(alice)
    with pytest.raises(LimitExceeded, match="uploaded 2 videos today"):
        limiter.use_upload(alice)
    limiter.use_pexels(alice)
    with pytest.raises(LimitExceeded, match="downloaded new videos for you 1 times"):
        limiter.use_pexels(alice)
    usage = limiter.usage("web:a")
    assert (usage.uploads_used, usage.pexels_used) == (2, 1)


def test_redis_down_means_no_limits_not_an_outage(limiter, redis_):
    redis_.down = True
    limiter.check_request(Principal(viewer="web:a", ip="1.1.1.1"))
    limiter.use_upload(Principal(viewer="web:a"))
    limiter.add_tokens(Principal(viewer="web:a"), 10)


def test_every_recorded_model_call_counts_against_whoever_it_was_for(limiter):
    recorder = UsageRecorder(MagicMock(), MagicMock(cost=lambda call: 0.0, reference="x"), "api", limiter)
    with acting_for(Principal(viewer="web:a", ip="1.1.1.1")):
        recorder.record(LLMCall("understand", "m", prompt_tokens=400, output_tokens=25), "accepted")
    recorder.record(LLMCall("caption", "m", prompt_tokens=200, output_tokens=30), "accepted")  # no one: global only
    assert limiter.usage("web:a").tokens_used == 425


def test_a_too_long_upload_explains_itself():
    reply = upload_failed("trip.mp4", "VideoTooLong: it's 14:20 long, and uploads can be up to 10:00")
    assert reply == "Sorry, “trip.mp4” is too long: it's 14:20 long, and uploads can be up to 10:00. Send a shorter clip."
