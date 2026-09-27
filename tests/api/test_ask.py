from src.dependencies import get_ask_service
from src.main import app
from src.services.answering import Answer, AnsweredClip
from src.services.clips import ClipRange
from src.services.search import SearchHit
from src.services.understanding import Intent, Understanding


class FakeAsk:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def ask(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.answer


def test_ask_returns_clip_url_answer_and_what_was_understood(client):
    doc = {"segment_id": "s1", "video_id": "v1", "kind": "speech", "start_sec": 0.0, "end_sec": 14.48, "video_title": "host_talk"}
    hit = SearchHit(doc, 1.0, 9.48, 10.86, 1.0, "AI is changing everything.", None)
    clip = AnsweredClip(
        hit, ClipRange(8.73, 11.61), "clips/v1/000008730-000011610.mp4", False, 'At 0:09–0:11 in "host_talk": "…"'
    )
    understood = Understanding(Intent(type="quote", phrase="AI is changing everything"), "llm", 1.2, notes=[])
    fake = FakeAsk(Answer(understood, "answered", "exact_phrase", clip.explanation, [clip], {"total": 2.0}))
    app.dependency_overrides[get_ask_service] = lambda: fake

    body = client.post("/api/v1/ask", json={"query": "where the host says AI is changing everything", "video_id": "v1"}).json()

    assert fake.calls[0][1]["video_id"] == "v1"
    assert body["status"] == "answered" and body["understood"]["intent"]["phrase"] == "AI is changing everything"
    assert body["clips"][0]["url"] == "http://s3.test/clips/v1/000008730-000011610.mp4"
    assert body["clips"][0]["duration_sec"] == 2.88


def test_ask_needs_subject_has_no_clips(client):
    understood = Understanding(Intent(type="visual", exclude=["people"]), "rules", 0.5, notes=["no subject"])
    app.dependency_overrides[get_ask_service] = lambda: FakeAsk(
        Answer(understood, "needs_subject", "none", "What would you like to see or hear?")
    )
    body = client.post("/api/v1/ask", json={"query": "exclude people"}).json()
    assert body["status"] == "needs_subject" and body["clips"] == []


def test_ask_validates(client):
    assert client.post("/api/v1/ask", json={"query": ""}).status_code == 422
    assert client.post("/api/v1/ask", json={"query": "x", "max_clips": 9}).status_code == 422
