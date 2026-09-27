from src.dependencies import get_ingestion_service
from src.main import app
from src.models import VideoSource, VideoStatus
from src.repositories import VideoRepository


def _seed(database):
    with database.get_session() as session:
        repo = VideoRepository(session)
        video = repo.create(source=VideoSource.UPLOAD, title="talk", s3_key="raw/x/source.mp4", status=VideoStatus.READY)
        repo.replace_segments(
            video.id,
            [
                {"kind": "visual", "idx": 0, "start_sec": 0.0, "end_sec": 5.0, "frame_key": "frames/x/000002500.jpg", "frame_time_sec": 2.5},
                {"kind": "speech", "idx": 0, "start_sec": 1.0, "end_sec": 3.0, "text": "AI is changing everything",
                 "words": [{"word": "AI", "start": 1.0, "end": 1.3, "prob": 0.9}]},
            ],
        )  # fmt: skip
        session.commit()
        return video.id


def test_upload_returns_202_and_queues(client, database, storage, queued):
    response = client.post("/api/v1/videos", files={"file": ("talk.mp4", b"fake-bytes", "video/mp4")})
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued" and body["source"] == "upload"
    assert queued == [("process", body["id"])]
    storage.upload_fileobj.assert_called_once()


def test_upload_rejects_non_video(client):
    response = client.post("/api/v1/videos", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_get_video_with_counts_and_url(client, database):
    video_id = _seed(database)
    body = client.get(f"/api/v1/videos/{video_id}").json()
    assert body["status"] == "ready"
    assert body["segment_counts"] == {"speech": 1, "visual": 1}
    assert body["video_url"] == "http://s3.test/raw/x/source.mp4"


def test_segments_filter_and_frame_urls(client, database):
    video_id = _seed(database)
    visual = client.get(f"/api/v1/videos/{video_id}/segments", params={"kind": "visual"}).json()["items"]
    assert len(visual) == 1 and visual[0]["frame_url"] == "http://s3.test/frames/x/000002500.jpg"
    speech = client.get(f"/api/v1/videos/{video_id}/segments", params={"kind": "speech"}).json()["items"]
    assert speech[0]["words"][0]["word"] == "AI"


def test_list_and_404(client, database):
    _seed(database)
    assert client.get("/api/v1/videos").json()["total"] == 1
    assert client.get("/api/v1/videos/does-not-exist").status_code == 404


def test_pexels_endpoint_maps_service_result(client, fake_services):
    class FakeResult:
        query, requested, queued, skipped_existing, skipped_unsuitable, pages_searched = "dog", 2, [], 2, 0, 1

    class FakeService:
        def ingest_pexels(self, pexels, query, count):
            assert (query, count) == ("dog", 2)
            return FakeResult()

    app.dependency_overrides[get_ingestion_service] = FakeService
    body = client.post("/api/v1/videos/pexels", json={"query": "dog", "count": 2}).json()
    assert body["skipped_existing"] == 2 and body["queued"] == []


def test_reprocess_requeues_failed_and_rejects_in_flight(client, database, monkeypatch):
    queued = []
    monkeypatch.setattr("src.dependencies._enqueue_process", lambda video_id: queued.append(video_id) or "job-1")
    with database.get_session() as session:
        repo = VideoRepository(session)
        failed = repo.create(source=VideoSource.UPLOAD, s3_key="raw/a/source.mp4", status=VideoStatus.FAILED, error="boom")
        busy = repo.create(source=VideoSource.UPLOAD, s3_key="raw/b/source.mp4", status=VideoStatus.PROCESSING)
        session.commit()

    response = client.post(f"/api/v1/videos/{failed.id}/reprocess")
    assert response.status_code == 202 and response.json()["status"] == "queued"
    assert queued == [failed.id]
    assert client.post(f"/api/v1/videos/{busy.id}/reprocess").status_code == 409
