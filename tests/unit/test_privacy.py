from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request

from src.config import Settings
from src.models import VideoSource, VideoStatus
from src.repositories import VideoRepository, can_see
from src.services.agent import ChatService, ConversationStore
from src.services.cache import CacheClient
from src.services.identity import COOKIE, sign, unsign, viewer_from_request
from src.services.ingestion import DeletionService, VideoBusy
from tests.unit.test_caching import FakeRedis

SETTINGS = Settings(_env_file=None, session_secret="s3cret", service_token="bot-token")


def request(headers: dict[str, str] | None = None, cookie: str | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if cookie:
        raw.append((b"cookie", f"{COOKIE}={cookie}".encode()))
    return Request({"type": "http", "headers": raw})


# ---- identity ----------------------------------------------------------------------------------------------------
def test_a_signed_session_cannot_be_forged():
    token = sign("abc", "s3cret")
    assert unsign(token, "s3cret") == "abc"
    assert unsign(token.replace("abc", "abd"), "s3cret") is None  # someone else's id, same signature
    assert unsign(token, "other-secret") is None
    assert unsign(None, "s3cret") is None and unsign("no-dot", "s3cret") is None


def test_viewer_comes_from_the_cookie_or_a_trusted_service():
    assert viewer_from_request(request(cookie=sign("abc", "s3cret")), SETTINGS) == "web:abc"
    assert viewer_from_request(request(cookie="abc.forged"), SETTINGS) is None
    bot = {"x-service-token": "bot-token", "x-user-id": "tg:42"}
    assert viewer_from_request(request(bot), SETTINGS) == "tg:42"
    # Without the right token the user id is ignored: anyone could claim to be tg:42
    assert viewer_from_request(request({"x-service-token": "guess", "x-user-id": "tg:42"}), SETTINGS) is None
    assert viewer_from_request(request({"x-user-id": "tg:42"}), SETTINGS) is None


# ---- visibility in Postgres --------------------------------------------------------------------------------------
def _video(repo: VideoRepository, owner: str | None, title: str, **fields):
    return repo.create(source=VideoSource.UPLOAD, title=title, status=VideoStatus.READY, owner_id=owner, **fields)


def test_viewers_see_the_library_and_only_their_own_uploads(database):
    with database.get_session() as session:
        repo = VideoRepository(session)
        library, mine, theirs = _video(repo, None, "library"), _video(repo, "web:a", "mine"), _video(repo, "web:b", "theirs")
        titles = lambda viewer: sorted(v.title for v in repo.list_videos(visible_to=viewer)[0])
        assert titles("web:a") == ["library", "mine"]
        assert titles(None) == ["library"]
        assert sorted(v.title for v in repo.list_videos(everything=True)[0]) == ["library", "mine", "theirs"]
        assert can_see(library, None) and can_see(mine, "web:a") and not can_see(theirs, "web:a")
        assert repo.get_visible(theirs.id, "web:a") is None  # exactly as if it didn't exist
        assert repo.owns_any("web:a") and not repo.owns_any("web:c") and not repo.owns_any(None)


def test_only_owned_uploads_expire(database):
    old = datetime.now(UTC) - timedelta(days=8)
    with database.get_session() as session:
        repo = VideoRepository(session)
        _video(repo, None, "old library", created_at=old)
        _video(repo, "web:a", "old upload", created_at=old)
        _video(repo, "web:a", "new upload")
        expired = repo.expired_uploads(datetime.now(UTC) - timedelta(days=7))
        assert [v.title for v in expired] == ["old upload"]


# ---- deleting ----------------------------------------------------------------------------------------------------
def test_deletion_removes_index_files_and_rows_then_invalidates_cached_answers(database):
    with database.get_session() as session:
        video_id = _video(VideoRepository(session), "web:a", "mine").id
        session.commit()
    opensearch, storage, versions = MagicMock(), MagicMock(), MagicMock()
    opensearch.delete_video.return_value, storage.delete_prefix.return_value = 12, 3

    deleted = DeletionService(database, storage, opensearch, versions).delete(video_id)

    assert (deleted.documents, deleted.files) == (12, 9)
    opensearch.delete_video.assert_called_once_with(video_id)
    assert [c.args[0] for c in storage.delete_prefix.call_args_list] == [
        f"raw/{video_id}/",
        f"frames/{video_id}/",
        f"clips/{video_id}/",
    ]
    versions.bump.assert_called_once()
    with database.get_session() as session:
        assert VideoRepository(session).get(video_id) is None


def test_a_video_being_processed_is_not_deleted(database):
    with database.get_session() as session:
        video = VideoRepository(session).create(source=VideoSource.UPLOAD, status=VideoStatus.PROCESSING, owner_id="web:a")
        session.commit()
    opensearch = MagicMock()
    with pytest.raises(VideoBusy):
        DeletionService(database, MagicMock(), opensearch, None).delete(video.id)
    opensearch.delete_video.assert_not_called()


# ---- the chat ------------------------------------------------------------------------------------------------------
@pytest.fixture
def chat():
    toolbox = MagicMock()
    toolbox.my_videos.return_value = [{"id": "v1", "title": "beach day", "status": "ready"}]
    toolbox.delete_video.return_value = "deleted"
    store = ConversationStore(CacheClient(FakeRedis()))
    return (lambda viewer: ChatService(toolbox, store, router=None, recorder=None, viewer=viewer)), toolbox


def test_delete_asks_first_and_deletes_only_on_yes(chat):
    service, toolbox = chat
    first = service("web:a").turn(None, "delete my video")
    assert "Delete “beach day”?" in first.reply
    toolbox.delete_video.assert_not_called()
    second = service("web:a").turn(first.conversation_id, "yes")
    assert "Deleted “beach day”" in second.reply
    toolbox.delete_video.assert_called_once_with("v1")


def test_no_keeps_the_video_and_the_question_expires(chat):
    service, toolbox = chat
    first = service("web:a").turn(None, "please remove my upload")
    assert service("web:a").turn(first.conversation_id, "no").reply == "Okay, I'll keep it."
    service("web:a").turn(first.conversation_id, "yes")  # no open question any more: nothing happens
    toolbox.delete_video.assert_not_called()


def test_nothing_to_delete_without_uploads(chat):
    service, toolbox = chat
    toolbox.my_videos.return_value = []
    assert "nothing of yours to delete" in service("web:a").turn(None, "delete all my videos").reply


def test_someone_elses_conversation_id_starts_a_new_conversation(chat):
    service, _ = chat
    first = service("web:a").turn(None, "delete my video")
    other = service("web:b").turn(first.conversation_id, "yes")
    assert other.conversation_id != first.conversation_id  # b can't confirm a's deletion, or read a's memory
    assert "Deleted" not in other.reply
