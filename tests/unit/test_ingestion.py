import io
from unittest.mock import MagicMock

import pytest

from src.models import VideoSource, VideoStatus
from src.repositories import VideoRepository
from src.services.ingestion import IngestionService
from src.services.pexels.models import PexelsSearchResponse
from tests.conftest import load_fixture


@pytest.fixture
def pexels():
    client = MagicMock()
    client.search_videos.return_value = PexelsSearchResponse.model_validate(load_fixture("pexels_search_dog.json"))
    client.rate_limit_remaining = 100
    return client


def _service(session, storage, settings, calls):
    def enqueue(kind):
        def _enqueue(video_id):
            calls.append((kind, video_id))
            return f"job-{video_id[:8]}"

        return _enqueue

    return IngestionService(session, storage, settings, enqueue_process=enqueue("process"), enqueue_download=enqueue("download"))


def test_upload_stores_bytes_first_then_queues_id(database, storage, settings):
    calls = []
    with database.get_session() as session:
        video = _service(session, storage, settings, calls).create_upload(io.BytesIO(b"x"), "talk.MP4", "video/mp4", 1)

    key = storage.upload_fileobj.call_args.args[1]
    assert key == f"raw/{video.id}/source.mp4"
    assert calls == [("process", video.id)]  # the queue only carries the id
    with database.get_session() as session:
        stored = VideoRepository(session).get(video.id)
        assert (stored.status, stored.source, stored.s3_key, stored.job_id) == (
            VideoStatus.QUEUED,
            VideoSource.UPLOAD,
            key,
            f"job-{video.id[:8]}",
        )


def test_pexels_ingest_filters_and_is_idempotent(database, storage, settings, pexels):
    calls = []
    with database.get_session() as session:
        first = _service(session, storage, settings, calls).ingest_pexels(pexels, "dog", count=5)

    # 240 s video exceeds PEXELS_MAX_DURATION_SEC=60 → unsuitable; the other two are queued
    assert [v.source_id for v in first.queued] == ["5340598", "2222222"]
    assert first.skipped_unsuitable == 1
    assert [kind for kind, _ in calls] == ["download", "download"]
    assert first.queued[0].source_file_url.endswith("hd_1280_720.mp4")
    assert first.queued[0].title == "Dogs with their tongues out"
    assert first.queued[0].author_name == "Dimitri Baret"

    with database.get_session() as session:
        second = _service(session, storage, settings, calls).ingest_pexels(pexels, "dog", count=5)
    assert second.queued == [] and second.skipped_existing == 2
    assert len(calls) == 2  # nothing re-queued


def test_enqueue_failure_marks_video_failed(database, storage, settings):
    def broken(video_id):
        raise ConnectionError("redis down")

    with database.get_session() as session:
        service = IngestionService(session, storage, settings, enqueue_process=broken, enqueue_download=broken)
        with pytest.raises(ConnectionError):
            service.create_upload(io.BytesIO(b"x"), "a.mp4", "video/mp4", 1)

    with database.get_session() as session:
        videos, _ = VideoRepository(session).list_videos()
        assert videos[0].status == VideoStatus.FAILED and "redis down" in videos[0].error


def test_replace_segments_is_idempotent(database):
    with database.get_session() as session:
        repo = VideoRepository(session)
        video = repo.create(source=VideoSource.UPLOAD, status=VideoStatus.PROCESSING)
        segs = [
            {"kind": "visual", "idx": 0, "start_sec": 0.0, "end_sec": 5.0},
            {"kind": "speech", "idx": 0, "start_sec": 0.0, "end_sec": 4.0, "text": "hi"},
        ]
        repo.replace_segments(video.id, segs)
        repo.replace_segments(video.id, segs)  # redelivered job
        session.commit()
        assert repo.segment_counts(video.id) == {"visual": 1, "speech": 1}
