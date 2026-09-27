from unittest.mock import patch

import pytest

HEALTHY = {"status": "healthy", "message": "ok"}


@pytest.fixture
def external_ok():
    with (
        patch("src.routers.ping._check_http", return_value=HEALTHY),
        patch("src.routers.ping._check_worker", return_value=HEALTHY),
    ):
        yield


def test_ping(client):
    response = client.get("/api/v1/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "pong"}


def test_health_all_healthy(client, external_ok):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["services"]) == {"database", "redis", "object_storage", "opensearch", "embedder", "ollama", "worker"}
    assert body["services"]["database"]["message"] == "Connected successfully"


def test_health_degraded_when_a_dependency_is_down(client, fake_services, external_ok):
    fake_services["cache_client"].health_check.return_value = {"status": "unhealthy", "message": "Redis ping failed"}
    fake_services["database"].healthcheck.return_value = {"status": "unavailable", "error": "connection refused"}

    body = client.get("/api/v1/health").json()

    assert body["status"] == "degraded"
    assert body["services"]["redis"] == {"status": "unhealthy", "message": "Redis ping failed"}
    assert body["services"]["database"] == {"status": "unhealthy", "message": "connection refused"}


def test_health_is_only_served_under_api_prefix(client):
    assert client.get("/health").status_code == 404
