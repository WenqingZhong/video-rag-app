from unittest.mock import MagicMock

import pytest

from src.services.agent import ChatService, Conversation, ConversationStore, ToolResult, TurnContext, decide, decide_with_rules
from src.services.agent.decide import RouterReply
from src.services.cache import CacheClient
from src.services.usage import LLMCall
from tests.unit.test_caching import FakeRedis


# ---- rules first -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("message", "context", "action"),
    [
        ("find this", TurnContext(image_attached=True), "find_by_image"),
        ("yes please", TurnContext(offered_fetch="horse"), "fetch_from_pexels"),
        ("no thanks", TurnContext(offered_fetch="horse"), "reply"),
        ("yes", TurnContext(), "reply"),
        ("download some horse videos from pexels", TurnContext(), "fetch_from_pexels"),
        ("hi", TurnContext(), "reply"),
        ("what videos do you have?", TurnContext(), "list_videos"),
    ],
)
def test_clear_patterns_are_decided_without_the_model(message, context, action):
    router = MagicMock()
    decision = decide(message, context, router)
    assert decision.action == action and decision.confident and not router.route.called


def test_open_ended_messages_go_to_the_model():
    router = MagicMock()
    router.route.return_value = (RouterReply(action="find_clip", text="a dog"), LLMCall("route", "qwen", 300, 20))
    decision = decide("give me a clip of a dog", TurnContext(), router)
    assert router.route.called and decision.source == "llm" and decision.text == "a dog"


def test_the_model_cannot_download_without_consent():
    router = MagicMock()
    router.route.return_value = (RouterReply(action="fetch_from_pexels", text="horse"), LLMCall("route", "qwen", 300, 20))
    decision = decide("a horse running", TurnContext(last_request="a dog", last_status="answered"), router)
    assert decision.action == "find_clip" and decision.source == "rules" and "no consent" in decision.notes[-1]


def test_an_answer_written_as_the_question_is_replaced_by_the_users_words():
    router = MagicMock()
    reply = RouterReply(action="answer_question", text="Deep sleep repairs tissue and helps memory.")
    router.route.return_value = (reply, LLMCall("route", "qwen", 300, 20))
    decision = decide("tell me what deep sleep is good for", TurnContext(), router)
    assert decision.action == "answer_question" and decision.text == "tell me what deep sleep is good for"


def test_follow_up_rules_combine_with_the_last_request():
    context = TurnContext(last_request="a beach", last_status="answered")
    assert decide_with_rules("now without people", context).text == "a beach without people"
    assert decide_with_rules("what about cats?", context).text == "cats"
    assert decide_with_rules("no, I meant a calm sea", context).text == "a calm sea"


# ---- memory --------------------------------------------------------------------------------------------------------
def test_conversation_is_remembered_and_redis_outage_starts_fresh():
    fake = FakeRedis()
    store = ConversationStore(CacheClient(fake))
    conversation = store.load(None)
    conversation.last_request = "a dog"
    store.save(conversation)
    assert store.load(conversation.id).last_request == "a dog"
    fake.down = True
    assert store.load(conversation.id).last_request is None  # no memory, but the turn still works


# ---- the graph, turn by turn -----------------------------------------------------------------------------------------
def clip(video_id: str, start: float) -> dict:
    return {"url": f"http://s3/{video_id}", "video_id": video_id, "title": video_id, "start_sec": start, "end_sec": start + 5,
            "explanation": f"{video_id} at {start}"}  # fmt: skip


@pytest.fixture
def chat():
    toolbox = MagicMock()
    store = ConversationStore(CacheClient(FakeRedis()))
    return ChatService(toolbox, store, router=None, recorder=None), toolbox  # rules only: deterministic


def test_a_no_match_offers_pexels_and_yes_downloads(chat):
    service, toolbox = chat
    toolbox.run.return_value = ToolResult(True, "No match", {"status": "no_match", "answer": "No moment matched footage of \"a horse\".",
                                                             "intent": {"visual": "a horse"}, "clips": []})  # fmt: skip
    first = service.turn(None, "a horse")
    assert "Pexels" in first.reply and first.action == "find_clip"

    toolbox.run.return_value = ToolResult(
        True, "Fetching", {"query": "a horse", "video_ids": ["v1", "v2"], "skipped_existing": 0}
    )
    second = service.turn(first.conversation_id, "yes please")
    assert second.action == "fetch_from_pexels" and second.decided_by == "rules" and second.fetching == ["v1", "v2"]
    assert toolbox.run.call_args.args == ("fetch_from_pexels", {"query": "a horse", "count": 3})


def test_an_offer_is_open_for_one_turn_only(chat):
    service, toolbox = chat
    toolbox.run.return_value = ToolResult(
        True, "No match", {"status": "no_match", "answer": "No match.", "intent": {"visual": "a horse"}, "clips": []}
    )
    first = service.turn(None, "a horse")
    toolbox.run.return_value = ToolResult(True, "Found", {"status": "answered", "answer": "dog", "clips": [clip("dog", 0)]})
    service.turn(first.conversation_id, "a dog")
    third = service.turn(first.conversation_id, "yes please")
    assert third.action == "reply"  # the horse offer expired: "yes" alone downloads nothing


def test_another_one_skips_clips_already_shown(chat):
    service, toolbox = chat
    toolbox.run.return_value = ToolResult(True, "Found", {"status": "answered", "answer": "a", "clips": [clip("dogA", 0)]})
    first = service.turn(None, "a dog")
    toolbox.run.return_value = ToolResult(
        True, "Found", {"status": "answered", "answer": "a", "clips": [clip("dogA", 0), clip("dogB", 3)]}
    )
    second = service.turn(first.conversation_id, "another one")
    assert [c["video_id"] for c in second.clips] == ["dogB"]
    assert toolbox.run.call_args.args[1] == {"request": "a dog", "max_clips": 3}


def test_a_question_about_that_clip_is_scoped_to_its_video(chat):
    service, toolbox = chat
    toolbox.run.return_value = ToolResult(True, "Found", {"status": "answered", "answer": "a", "clips": [clip("perro", 0)]})
    first = service.turn(None, "a dog")
    toolbox.run.return_value = ToolResult(True, "Brown.", {"status": "answered", "citations": [], "clip_url": None})
    service.turn(first.conversation_id, "what colour is the dog in that clip?")
    assert toolbox.run.call_args.args == (
        "answer_question",
        {"question": "what colour is the dog in that clip?", "video_id": "perro"},
    )


def test_a_failed_tool_is_an_apology_not_a_crash(chat):
    service, toolbox = chat
    toolbox.run.return_value = ToolResult(False, "find_clip failed: ConnectionError")
    assert service.turn(None, "a dog").reply.startswith("Sorry, that didn't work")


def test_conversation_keeps_its_state_object():
    assert Conversation(id="x").context(image_attached=True).image_attached


def test_a_follow_up_the_model_passes_on_alone_is_combined_with_the_last_request():
    router = MagicMock()
    router.route.return_value = (RouterReply(action="find_clip", text="now without people"), LLMCall("route", "qwen", 300, 20))
    decision = decide("now without people", TurnContext(last_request="a beach", last_status="answered"), router)
    assert decision.text == "a beach without people" and "dropped the last request" in decision.notes[-1]


def test_a_new_request_is_not_narrowed_to_the_last_clip():
    router = MagicMock()
    router.route.return_value = (
        RouterReply(action="find_clip", text="a beach", about_last_video=True),
        LLMCall("route", "q", 300, 20),
    )
    context = TurnContext(last_request="a dog", last_status="answered", last_video="Dogs relaxing")
    assert decide("a beach", context, router).about_last_video is False
    router.route.return_value = (RouterReply(action="answer_question", text="what colour is the dog in that clip?",
                                             about_last_video=True), LLMCall("route", "q", 300, 20))  # fmt: skip
    assert decide("what colour is the dog in that clip?", context, router).about_last_video is True


def test_a_refined_request_may_show_the_same_clip_again(chat):
    service, toolbox = chat
    toolbox.run.return_value = ToolResult(True, "Found", {"status": "answered", "answer": "a", "clips": [clip("beach", 7)]})
    first = service.turn(None, "a beach")
    second = service.turn(first.conversation_id, "now without people")  # the empty beach is the same clip
    assert [c["video_id"] for c in second.clips] == ["beach"]


# ---- after "yes": waiting for the downloaded videos ------------------------------------------------------------------
def fetch_flow(chat, statuses: dict, retry: dict):
    """Offer → yes → poll, with the given video statuses and the retried search's result."""
    service, toolbox = chat
    tools = {
        "find_clip": [
            ToolResult(
                True,
                "No match",
                {"status": "no_match", "answer": "No moment matched.", "intent": {"visual": "a horse"}, "clips": []},
            ),
            ToolResult(True, "Found", retry),
        ],
        "fetch_from_pexels": [
            ToolResult(True, "Fetching", {"query": "a horse", "video_ids": list(statuses), "skipped_existing": 0})
        ],
    }
    ready = sum(s == "ready" for s in statuses.values())
    failed = sum(s == "failed" for s in statuses.values())
    check = ToolResult(
        True, "", {"statuses": statuses, "ready": ready, "pending": len(statuses) - ready - failed, "failed": failed}
    )
    toolbox.run.side_effect = lambda name, args: check if name == "check_videos" else tools[name].pop(0)
    first = service.turn(None, "a horse")
    service.turn(first.conversation_id, "yes please")
    return service, first.conversation_id, toolbox


def test_nothing_downloading_is_idle(chat):
    service, _ = chat
    assert service.poll("unknown").status == "idle"


def test_still_processing_reports_progress(chat):
    service, cid, _ = fetch_flow(chat, {"v1": "ready", "v2": "processing"}, {})
    update = service.poll(cid)
    assert update.status == "pending" and update.progress == {"ready": 1, "pending": 1, "failed": 0} and update.reply is None


def test_once_processed_the_original_request_is_retried(chat):
    retry = {"status": "answered", "answer": "horse", "clips": [clip("horse1", 2)]}
    service, cid, toolbox = fetch_flow(chat, {"v1": "ready", "v2": "failed"}, retry)
    update = service.poll(cid)
    assert update.status == "done" and update.reply.startswith("Your “a horse” videos are ready.")
    assert [c["video_id"] for c in update.clips] == ["horse1"]
    assert ("find_clip", {"request": "a horse", "max_clips": 3}) == toolbox.run.call_args.args
    assert service.poll(cid).status == "idle"  # delivered once


def test_processed_but_no_match_says_so(chat):
    retry = {"status": "no_match", "answer": "No moment matched.", "intent": {"visual": "a horse"}, "clips": []}
    service, cid, _ = fetch_flow(chat, {"v1": "ready"}, retry)
    update = service.poll(cid)
    assert update.status == "done" and "none matched" in update.reply


def test_waiting_gives_up_after_the_timeout(chat, monkeypatch):
    service, cid, _ = fetch_flow(chat, {"v1": "processing"}, {})
    service.graph.fetch_timeout_sec = 0
    monkeypatch.setattr("src.services.agent.graph.time.time", lambda: 10**10)
    update = service.poll(cid)
    assert update.status == "timed_out" and "taking too long" in update.reply


def test_empty_polls_are_not_traced():
    from src.services.tracing import discard_trace, span, start_trace

    saved = []
    with start_trace("GET /api/v1/chat/{conversation_id}/updates", service="api", save=saved.extend), span("x"):
        discard_trace()
    assert saved == []


# ---- follow-ups that repeat the last kind of request ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("message", "last_action", "action", "text"),
    [
        ("what about frying?", "find_clip", "find_clip", "frying"),
        ("and a hot air balloon", "find_clip", "find_clip", "a hot air balloon"),
        ("what about naps?", "answer_question", "answer_question", "what about naps?"),
    ],
)
def test_a_new_subject_repeats_the_last_kind_of_request(message, last_action, action, text):
    context = TurnContext(last_request="something", last_status="answered", last_action=last_action)
    decision = decide(message, context, MagicMock())
    assert (decision.action, decision.text, decision.confident) == (action, text, True)


def test_and_one_with_still_refines():
    context = TurnContext(last_request="people in a meeting", last_status="answered", last_action="find_clip")
    assert decide_with_rules("and one with a laptop", context).text == "people in a meeting with a laptop"


def test_show_me_that_part_after_an_answer_shows_its_moment():
    context = TurnContext(last_request="what does melatonin do", last_status="answered", last_action="answer_question",
                          last_video="sleep_talk")  # fmt: skip
    decision = decide("show me that part", context, MagicMock())
    assert (decision.action, decision.text, decision.about_last_video) == ("find_clip", "what does melatonin do", True)


def test_a_model_request_that_ignores_the_message_is_replaced():
    router = MagicMock()
    router.route.return_value = (RouterReply(action="find_clip", text="a unicorn"), LLMCall("route", "q", 300, 20))
    context = TurnContext(last_request="a unicorn", last_status="no_match", last_action="find_clip")
    decision = decide("show me cats instead", context, router)
    assert decision.text == "show me cats instead" and "ignores the message" in decision.notes[-1]


# ---- uploading a video in the chat -------------------------------------------------------------------------------------
def test_an_uploaded_video_becomes_the_focus_when_ready(chat):
    service, toolbox = chat
    toolbox.upload_video.return_value = "up1"
    first = service.turn(None, "", video=(b"bytes", "talk.mp4", "video/mp4", 5))
    assert first.uploading == "up1" and "Processing" in first.reply

    toolbox.video_status.return_value = {"status": "processing"}
    assert service.poll(first.conversation_id).status == "pending"

    toolbox.video_status.return_value = {"status": "ready", "title": "talk", "duration_sec": 102.0, "speech": True}
    ready = service.poll(first.conversation_id)
    assert ready.status == "done" and "“talk.mp4” (1:42) is ready" in ready.reply

    toolbox.run.return_value = ToolResult(True, "Found", {"status": "answered", "answer": "a", "clips": [clip("up1", 9)]})
    service.turn(first.conversation_id, "the part where she says sleep well")
    assert toolbox.run.call_args.args[1]["video_id"] == "up1"  # the user's video is searched first


def test_a_request_sent_with_the_upload_is_answered_when_ready(chat):
    service, toolbox = chat
    toolbox.upload_video.return_value = "up1"
    first = service.turn(None, "the part where he says hello", video=(b"x", "v.mp4", "video/mp4", 1))
    assert "Then I'll find “the part where he says hello”" in first.reply
    toolbox.video_status.return_value = {"status": "ready", "title": "v", "duration_sec": 20.0, "speech": True}
    toolbox.run.return_value = ToolResult(True, "Found", {"status": "answered", "answer": "a", "clips": [clip("up1", 3)]})
    update = service.poll(first.conversation_id)
    assert update.status == "done" and update.clips[0]["video_id"] == "up1"
    assert toolbox.run.call_args.args == ("find_clip", {"request": "the part where he says hello", "video_id": "up1"})


def test_not_in_the_upload_just_says_so(chat):
    service, toolbox = chat
    toolbox.upload_video.return_value = "up1"
    first = service.turn(None, "", video=(b"x", "v.mp4", "video/mp4", 1))
    toolbox.video_status.return_value = {"status": "ready", "title": "v", "duration_sec": 20.0, "speech": False}
    service.poll(first.conversation_id)
    toolbox.run.return_value = ToolResult(
        True, "No match", {"status": "no_match", "answer": "No match.", "intent": {"visual": "a dog"}, "clips": []}
    )
    turn = service.turn(first.conversation_id, "a dog")
    assert turn.reply == "I couldn't find “a dog” in your video “v.mp4”." and turn.clips == []
    assert toolbox.run.call_count == 1  # no library search, and no Pexels offer
    assert service.turn(first.conversation_id, "yes please").action == "reply"


def test_a_request_sent_while_processing_waits_for_the_video(chat):
    service, toolbox = chat
    toolbox.upload_video.return_value = "up1"
    first = service.turn(None, "", video=(b"x", "talk.mp4", "video/mp4", 1))
    assert "you can ask now" in first.reply

    toolbox.video_status.return_value = {"status": "processing", "stage": "captioning 3/11"}
    waiting = service.turn(first.conversation_id, "the part where he says hello")
    assert "still processing" in waiting.reply and "describing the frames (3 of 11)" in waiting.reply
    assert not toolbox.run.called  # nothing searched yet

    pending = service.poll(first.conversation_id)
    assert pending.status == "pending" and pending.status_text.startswith("Step 6 of 9: describing the frames (3 of 11)")

    toolbox.video_status.return_value = {"status": "ready", "title": "talk", "duration_sec": 30.0, "speech": True}
    toolbox.run.return_value = ToolResult(True, "Found", {"status": "answered", "answer": "a", "clips": [clip("up1", 4)]})
    done = service.poll(first.conversation_id)
    assert done.status == "done" and done.reply.startswith("Your video “talk.mp4” (0:30) is ready.")
    assert toolbox.run.call_args.args == ("find_clip", {"request": "the part where he says hello", "video_id": "up1"})


def test_waiting_in_line_is_explained():
    from src.services.agent.replies import processing_status

    assert processing_status("queued", None, 75).startswith("Waiting in line")
    assert processing_status("processing", "transcribing", 5) == "Step 7 of 9: transcribing the speech · 0:05"


def test_a_failed_upload_says_so(chat):
    service, toolbox = chat
    toolbox.upload_video.return_value = "up1"
    first = service.turn(None, "", video=(b"x", "v.mp4", "video/mp4", 1))
    toolbox.video_status.return_value = {"status": "failed", "error": "ffprobe: invalid data"}
    update = service.poll(first.conversation_id)
    assert update.status == "done" and "couldn't process" in update.reply
