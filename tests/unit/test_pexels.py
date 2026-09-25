import httpx
import pytest

from src.services.pexels import PexelsClient, PexelsError
from src.services.pexels.models import PexelsSearchResponse
from tests.conftest import load_fixture


@pytest.fixture
def search() -> PexelsSearchResponse:
    return PexelsSearchResponse.model_validate(load_fixture("pexels_search_dog.json"))


def test_title_from_url_slug(search):
    assert search.videos[0].title == "Dogs with their tongues out"


def test_best_file_prefers_largest_within_height(search):
    assert search.videos[0].best_file(max_height=720).height == 720


def test_best_file_falls_back_to_smallest_when_all_too_large(search):
    assert search.videos[2].best_file(max_height=720).height == 1440


def _client(handler) -> PexelsClient:
    client = PexelsClient(api_key="test-key", max_retries=2)
    client.http = httpx.Client(base_url="https://api.pexels.com", transport=httpx.MockTransport(handler))
    return client


def test_client_retries_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("src.services.pexels.client.time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=load_fixture("pexels_search_dog.json"), headers={"x-ratelimit-remaining": "42"})

    client = _client(handler)
    result = client.search_videos("dog", per_page=3)
    assert len(calls) == 2 and len(result.videos) == 3
    assert calls[1].url.params["query"] == "dog"
    assert client.rate_limit_remaining == 42


def test_client_raises_on_auth_error():
    client = _client(lambda request: httpx.Response(401, text="unauthorized"))
    with pytest.raises(PexelsError, match="401"):
        client.search_videos("dog")


def test_client_requires_key():
    with pytest.raises(PexelsError):
        PexelsClient(api_key="")
