from fastapi.testclient import TestClient

from src.dependencies import get_deletion_service
from src.main import app
from src.models import VideoSource, VideoStatus
from src.repositories import VideoRepository


def _library_video(database) -> str:
    with database.get_session() as session:
        video = VideoRepository(session).create(source=VideoSource.PEXELS, title="stock", status=VideoStatus.READY)
        session.commit()
        return video.id


def test_uploads_are_private_to_their_uploader(client, database, queued):
    library_id = _library_video(database)
    alice, bob = client, TestClient(app)  # two browsers: each gets its own session cookie
    uploaded = alice.post("/api/v1/videos", files={"file": ("mine.mp4", b"x", "video/mp4")}).json()

    assert alice.get(f"/api/v1/videos/{uploaded['id']}").status_code == 200
    assert bob.get(f"/api/v1/videos/{uploaded['id']}").status_code == 404
    assert {v["id"] for v in alice.get("/api/v1/videos").json()["items"]} == {uploaded["id"], library_id}
    assert {v["id"] for v in bob.get("/api/v1/videos").json()["items"]} == {library_id}


def test_only_the_owner_can_delete_and_the_library_stays(client, database, queued):
    library_id = _library_video(database)
    deletion = type("Deletion", (), {"deleted": [], "delete": lambda self, vid: self.deleted.append(vid)})()
    app.dependency_overrides[get_deletion_service] = lambda: deletion
    alice, bob = client, TestClient(app)
    uploaded = alice.post("/api/v1/videos", files={"file": ("mine.mp4", b"x", "video/mp4")}).json()

    assert bob.delete(f"/api/v1/videos/{uploaded['id']}").status_code == 404
    assert alice.delete(f"/api/v1/videos/{library_id}").status_code == 404
    assert alice.delete(f"/api/v1/videos/{uploaded['id']}").status_code == 204
    assert deletion.deleted == [uploaded["id"]]
