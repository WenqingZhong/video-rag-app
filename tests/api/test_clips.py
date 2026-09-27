from unittest.mock import MagicMock

from src.dependencies import get_clip_service
from src.main import app
from src.models import VideoSource, VideoStatus
from src.repositories import VideoRepository


def _video(database, duration=17.44):
    with database.get_session() as session:
        video = VideoRepository(session).create(
            source=VideoSource.UPLOAD, s3_key="raw/x/source.mp4", status=VideoStatus.READY, duration_sec=duration
        )
        session.commit()
        return video.id


def test_clip_endpoint_cuts_clamps_and_returns_url(client, database):
    video_id = _video(database)
    service = MagicMock()
    service.get_or_cut.return_value = (f"clips/{video_id}/000008730-000017440.mp4", False)
    app.dependency_overrides[get_clip_service] = lambda: service

    body = client.post("/api/v1/clips", json={"video_id": video_id, "start_sec": 8.73, "end_sec": 99}).json()

    clip = service.get_or_cut.call_args.args[1]
    assert (clip.start_sec, clip.end_sec) == (8.73, 17.44)  # end clamped to the video's duration
    assert body["url"].startswith("http://s3.test/clips/") and body["cached"] is False and body["duration_sec"] == 8.71


def test_clip_endpoint_validates(client, database):
    video_id = _video(database)
    app.dependency_overrides[get_clip_service] = lambda: MagicMock()
    assert client.post("/api/v1/clips", json={"video_id": "nope", "start_sec": 0, "end_sec": 1}).status_code == 404
    assert client.post("/api/v1/clips", json={"video_id": video_id, "start_sec": 20, "end_sec": 30}).status_code == 422
