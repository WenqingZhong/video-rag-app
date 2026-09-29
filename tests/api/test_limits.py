import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.main import app
from src.services.cache import CacheClient
from src.services.limits import Limiter
from tests.unit.test_caching import FakeRedis

ADMIN = {"x-admin-token": Settings(_env_file=None).admin_token}


@pytest.fixture
def limiter(client, monkeypatch):
    limiter = Limiter(CacheClient(FakeRedis()), Settings(_env_file=None, limit_tokens_per_day=100, limit_uploads_per_day=1))
    monkeypatch.setattr(app.state, "limiter", limiter, raising=False)
    return limiter


def test_admin_endpoints_need_the_admin_token(client):
    assert client.post("/api/v1/admin/reindex").status_code == 403
    assert client.get("/api/v1/traces").status_code == 403
    assert client.post("/api/v1/videos/pexels", json={"query": "dog"}).status_code == 403


def test_out_of_tokens_is_a_429_with_the_reason(client, limiter):
    client.get("/app")
    viewer = "web:" + client.cookies.get("vr_session").rpartition(".")[0]
    limiter.add_tokens(type("P", (), {"viewer": viewer, "ip": None})(), 100)
    response = client.post("/api/v1/chat", data={"message": "a dog"})
    assert response.status_code == 429
    assert "today's allowance of 100 tokens" in response.json()["detail"] and "Retry-After" in response.headers
    assert client.get("/api/v1/me/usage").json()["tokens_left"] == 0
    other = TestClient(app)  # another person (and, in tests, the same address: its allowance is separate)
    assert other.get("/api/v1/me/usage").json()["tokens_left"] == 100


def test_uploads_per_day_and_admin_uploads_join_the_library(client, limiter, queued):
    upload = lambda headers=None: client.post(
        "/api/v1/videos", files={"file": ("a.mp4", b"x", "video/mp4")}, headers=headers or {}
    )
    assert upload().status_code == 202
    assert upload().status_code == 429
    library = upload(ADMIN)
    assert library.status_code == 202  # not counted, not limited
    assert TestClient(app).get(f"/api/v1/videos/{library.json()['id']}").status_code == 200  # visible to everyone
