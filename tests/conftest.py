from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.main import app


def _healthy(message: str = "ok") -> MagicMock:
    return MagicMock(return_value={"status": "healthy", "message": message})


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def fake_services(settings):
    """Stand-ins for everything the lifespan would normally create."""
    database = MagicMock()
    database.healthcheck = MagicMock(return_value={"status": "healthy", "database": "video_rag"})
    cache = MagicMock()
    cache.health_check = _healthy("redis ok")
    storage = MagicMock()
    storage.health_check = _healthy("storage ok")
    return {"settings": settings, "database": database, "cache_client": cache, "storage_client": storage}


@pytest.fixture
def client(fake_services):
    # No `with` block: the real lifespan (which connects to infra) is skipped.
    for name, value in fake_services.items():
        setattr(app.state, name, value)
    yield TestClient(app)
