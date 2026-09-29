from unittest.mock import MagicMock

import httpx
import pytest

from src.bot.telegram import HELP, TelegramBot


@pytest.fixture
def bot():
    b = TelegramBot("TOKEN", "http://api", storage=MagicMock(), allowed_chat_ids=None)
    b.call = MagicMock(return_value={})
    b.api = MagicMock()
    return b


def api_reply(body: dict):
    response = MagicMock()
    response.raise_for_status.return_value.json.return_value = body
    return response


def sent(bot, method: str) -> list[dict]:
    return [c.kwargs for c in bot.call.call_args_list if c.args[0] == method]


def test_start_and_new(bot):
    bot.handle({"chat": {"id": 1}, "text": "/start"})
    bot.conversations[1] = "tg-1-old"
    bot.handle({"chat": {"id": 1}, "text": "/new"})
    assert sent(bot, "sendMessage")[0]["text"] == HELP and 1 not in bot.conversations


def test_a_message_goes_to_the_chat_api_in_one_conversation(bot):
    bot.api.post.return_value = api_reply({"conversation_id": "c", "reply": "Hi!", "clips": [], "fetching": []})
    bot.handle({"chat": {"id": 7}, "text": "hi"})
    bot.handle({"chat": {"id": 7}, "text": "a dog"})
    ids = {c.kwargs["data"]["conversation_id"] for c in bot.api.post.call_args_list}
    assert len(ids) == 1 and ids.pop().startswith("tg-7-")
    assert sent(bot, "sendMessage")[0] == {"chat_id": 7, "text": "Hi!"}


def test_clips_are_uploaded_from_storage(bot):
    bot.storage.get_bytes.return_value = b"mp4 bytes"
    clip = {"url": "http://localhost:8333/x", "key": "clips/v/1.mp4", "explanation": "A dog."}
    bot.api.post.return_value = api_reply({"conversation_id": "c", "reply": "A dog.", "clips": [clip], "fetching": []})
    bot.handle({"chat": {"id": 7}, "text": "a dog"})
    [video] = sent(bot, "sendVideo")
    assert video["caption"] == "A dog." and video["files"]["video"][1] == b"mp4 bytes"
    assert sent(bot, "sendMessage") == []  # the caption already says it


def test_an_answer_is_sent_as_text_then_its_clip(bot):
    bot.storage.get_bytes.return_value = b"mp4"
    clip = {"url": "u", "key": "k", "explanation": "The moment this answer comes from."}
    bot.api.post.return_value = api_reply({"conversation_id": "c", "reply": "About 90 minutes.", "clips": [clip], "fetching": []})
    bot.handle({"chat": {"id": 7}, "text": "how long is a sleep cycle?"})
    assert sent(bot, "sendMessage")[0]["text"] == "About 90 minutes." and len(sent(bot, "sendVideo")) == 1


def test_a_private_bot_turns_strangers_away(bot):
    bot.allowed = {1}
    bot.handle({"chat": {"id": 2}, "text": "a dog"})
    assert "private" in sent(bot, "sendMessage")[0]["text"] and not bot.api.post.called


def test_the_api_being_down_is_an_apology(bot):
    bot.api.post.side_effect = httpx.ConnectError("down")
    bot.handle({"chat": {"id": 7}, "text": "a dog"})
    assert "went wrong" in sent(bot, "sendMessage")[0]["text"]


def test_one_status_message_is_edited_while_processing_then_the_result_arrives(bot):
    bot.storage.get_bytes.return_value = b"mp4"
    bot.call.side_effect = lambda method, **kw: {"message_id": 42} if method == "sendMessage" else {}
    clip = {"url": "u", "key": "k", "explanation": "A balloon."}
    bot.api.get.side_effect = [
        api_reply({"status": "pending", "status_text": "Step 6 of 9: describing the frames (3 of 11) · 0:42"}),
        api_reply({"status": "pending", "status_text": "Step 7 of 9: transcribing the speech · 0:55"}),
        api_reply({"status": "done", "reply": "Your video is ready. A balloon.", "clips": [clip]}),
    ]
    bot.wait_for_download(7, "c", every_sec=0)
    edits = [c.kwargs["text"] for c in bot.call.call_args_list if c.args[0] == "editMessageText"]
    assert edits[:2] == [
        "⏳ Step 6 of 9: describing the frames (3 of 11) · 0:42",
        "⏳ Step 7 of 9: transcribing the speech · 0:55",
    ]
    assert edits[2].startswith("✅ Processed in")
    assert all(c.kwargs.get("message_id") == 42 for c in bot.call.call_args_list if c.args[0] == "editMessageText")
    assert len(sent(bot, "sendVideo")) == 1 and "c" not in bot.watching


def test_a_request_waiting_for_the_video_starts_one_watcher(bot):
    bot.wait_for_download = MagicMock()
    bot.api.post.return_value = api_reply({"conversation_id": "c", "reply": "Your video is still processing…", "clips": [],
                                           "uploading": "v1"})  # fmt: skip
    bot.handle({"chat": {"id": 7}, "text": "the part where he says hello"})
    bot.handle({"chat": {"id": 7}, "text": "and the goodbye"})
    import time

    time.sleep(0.05)
    assert bot.wait_for_download.call_count == 1  # already watching that conversation


def test_the_token_never_reaches_the_logs(caplog):
    import logging

    from src.bot.telegram import RedactToken

    logger = logging.getLogger("redact-test")
    caplog.handler.addFilter(RedactToken("123:SECRET"))
    try:
        raise RuntimeError("GET https://api.telegram.org/file/bot123:SECRET/photos/x.jpg failed")
    except RuntimeError:
        logger.exception("download failed for %s", "https://api.telegram.org/bot123:SECRET/getFile")
    assert "SECRET" not in caplog.text and "<token>" in caplog.text


def test_a_video_is_uploaded_with_its_caption_as_the_request(bot):
    bot.file_bytes = MagicMock(return_value=b"mp4")
    bot.wait_for_download = MagicMock()
    bot.api.post.return_value = api_reply({"conversation_id": "c", "reply": "Got your video.", "clips": [], "uploading": "v1"})
    video = {"file_id": "f1", "file_size": 3_000_000, "file_name": "talk.mp4", "mime_type": "video/mp4"}
    bot.handle({"chat": {"id": 7}, "video": video, "caption": "the part where he says hello"})
    request = bot.api.post.call_args.kwargs
    assert request["data"]["message"] == "the part where he says hello" and request["files"]["video"][0] == "talk.mp4"


def test_a_video_over_20_mb_is_refused_politely(bot):
    bot.handle({"chat": {"id": 7}, "document": {"file_id": "f", "file_size": 30_000_000, "mime_type": "video/mp4"}})
    assert "over 20 MB" in sent(bot, "sendMessage")[0]["text"] and not bot.api.post.called


def test_the_greeting_leads_with_uploading_a_video():
    assert HELP.index("Got a video?") < HELP.index("Need some footage?") and "20 MB" in HELP


def test_each_user_is_sent_as_themselves_and_limits_are_shown_in_words(bot):
    limited = httpx.Response(429, json={"detail": "You've used today's allowance of 50,000 tokens."},
                             request=httpx.Request("POST", "http://api/api/v1/chat"))  # fmt: skip
    bot.api.post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429", request=limited.request, response=limited
    )
    bot.handle({"chat": {"id": 7}, "from": {"id": 42}, "text": "a dog"})
    assert bot.api.post.call_args.kwargs["headers"] == {"x-user-id": "tg:42"}
    assert sent(bot, "sendMessage")[0]["text"] == "You've used today's allowance of 50,000 tokens."


def test_usage_command(bot):
    bot.api.get.return_value = api_reply({"tokens_used": 1234, "tokens_limit": 50000, "tokens_left": 48766, "uploads_used": 1,
                                          "uploads_limit": 5, "pexels_used": 0, "pexels_limit": 3, "resets_in_sec": 3 * 3600 + 120})  # fmt: skip
    bot.handle({"chat": {"id": 7}, "from": {"id": 42}, "text": "/usage"})
    assert sent(bot, "sendMessage")[0]["text"] == (
        "Today you've used 1,234 of your 50,000 tokens (48,766 left), 1 of 5 video uploads, and 0 of 3 stock-footage "
        "downloads. Everything resets in 3 h 2 min."
    )
