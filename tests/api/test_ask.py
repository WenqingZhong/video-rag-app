from src.dependencies import get_ask_service
from src.main import app
from src.services.answering import Answer, AnsweredClip
from src.services.clips import ClipRange
from src.services.search import SearchHit
from src.services.understanding import Intent, Understanding
from src.services.usage import LLMCall


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
    understood.llm_call, understood.llm_outcome = LLMCall("understand", "qwen", 423, 26), "accepted"
    understood.cost_usd = 0.000553
    fake = FakeAsk(Answer(understood, "answered", "exact_phrase", clip.explanation, [clip], {"total": 2.0}))
    app.dependency_overrides[get_ask_service] = lambda: fake

    body = client.post("/api/v1/ask", json={"query": "where the host says AI is changing everything", "video_id": "v1"}).json()

    assert fake.calls[0][1]["video_id"] == "v1"
    assert body["status"] == "answered" and body["understood"]["intent"]["phrase"] == "AI is changing everything"
    assert body["clips"][0]["url"] == "http://s3.test/clips/v1/000008730-000011610.mp4"
    assert body["clips"][0]["duration_sec"] == 2.88
    assert body["usage"] == {
        "prompt_tokens": 423, "output_tokens": 26, "total_tokens": 449, "cost_usd": 0.000553, "outcome": "accepted",
        "saved_tokens": 0, "saved_cost_usd": 0.0,
    }  # fmt: skip


def test_ask_needs_subject_has_no_clips(client):
    understood = Understanding(Intent(type="visual", exclude=["people"]), "rules", 0.5, notes=["no subject"])
    app.dependency_overrides[get_ask_service] = lambda: FakeAsk(
        Answer(understood, "needs_subject", "none", "What would you like to see or hear?")
    )
    body = client.post("/api/v1/ask", json={"query": "exclude people"}).json()
    assert body["status"] == "needs_subject" and body["clips"] == []
    assert body["usage"]["total_tokens"] == 0 and body["usage"]["outcome"] is None  # rules only: no tokens


def test_ask_validates(client):
    assert client.post("/api/v1/ask", json={"query": ""}).status_code == 422
    assert client.post("/api/v1/ask", json={"query": "x", "max_clips": 9}).status_code == 422


def test_chat_turn_returns_reply_and_conversation_id(client):
    from src.dependencies import get_chat_service
    from src.services.agent import ChatTurn

    class FakeChat:
        def turn(self, conversation_id, message, image_key=None, video=None):
            return ChatTurn(conversation_id or "c1", f"you said {message}", "reply", "rules", seconds=0.01)

    app.dependency_overrides[get_chat_service] = FakeChat
    body = client.post("/api/v1/chat", data={"message": "hi"}).json()
    assert body["conversation_id"] == "c1" and body["reply"] == "you said hi" and body["decided_by"] == "rules"
    assert client.post("/api/v1/chat", data={"message": " "}).status_code == 422  # nothing to answer
