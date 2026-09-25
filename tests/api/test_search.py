from src.dependencies import get_search_service
from src.main import app
from src.services.search import ParsedQuery, SearchHit, SearchResult


class FakeSearch:
    def __init__(self, result):
        self.result, self.calls = result, []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.result


def _hit():
    source = {
        "segment_id": "s1", "video_id": "v1", "kind": "speech", "start_sec": 0.0, "end_sec": 14.5,
        "text": "... AI is changing everything ...", "video_title": "host_talk", "video_source": "upload",
        "video_s3_key": "raw/v1/source.mp4",
    }  # fmt: skip
    return SearchHit(
        source=source,
        score=7.1,
        match_start_sec=9.48,
        match_end_sec=10.84,
        match_score=1.0,
        matched_text="AI is changing everything.",
        highlight="<em>AI</em>",
    )


def test_search_returns_exact_moment_and_play_url(client):
    fake = FakeSearch(
        SearchResult(parsed=ParsedQuery("q", "AI is changing everything", "q"), strategy="exact_phrase", hits=[_hit()])
    )
    app.dependency_overrides[get_search_service] = lambda: fake

    body = client.post("/api/v1/search", json={"query": "host says ‘AI is changing everything’", "video_id": "v1"}).json()

    assert fake.calls[0][1]["video_id"] == "v1"
    assert body["phrase"] == "AI is changing everything" and body["strategy"] == "exact_phrase" and body["total"] == 1
    hit = body["hits"][0]
    assert (hit["match_start_sec"], hit["match_end_sec"]) == (9.48, 10.84)
    assert hit["play_url"] == "http://s3.test/raw/v1/source.mp4#t=8.48,11.84"  # 1 s padding each side
    assert hit["video"]["title"] == "host_talk"


def test_search_validates_input(client):
    assert client.post("/api/v1/search", json={"query": ""}).status_code == 422
    assert client.post("/api/v1/search", json={"query": "x", "kind": "audio"}).status_code == 422
