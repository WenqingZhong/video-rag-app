import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.models  # noqa: F401 - register tables
from src.config import Settings
from src.db.interfaces.postgresql import Base
from src.main import app

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _healthy(message: str = "ok") -> MagicMock:
    return MagicMock(return_value={"status": "healthy", "message": message})


class SQLiteDatabase:
    """In-memory stand-in for PostgreSQLDatabase (same get_session/healthcheck surface)."""

    def __init__(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def get_session(self):
        session = self.session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def healthcheck(self):
        return {"status": "healthy", "database": "sqlite"}

    def teardown(self):
        self.engine.dispose()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def database():
    db = SQLiteDatabase()
    yield db
    db.teardown()


@pytest.fixture
def storage():
    fake = MagicMock()
    fake.health_check = _healthy("storage ok")
    fake.presigned_url.side_effect = lambda key, expires_in=None: f"http://s3.test/{key}"
    return fake


@pytest.fixture
def fake_services(settings, database, storage):
    """Stand-ins for everything the lifespan would normally create."""
    db_mock = MagicMock(wraps=database)
    db_mock.healthcheck = MagicMock(return_value={"status": "healthy", "database": "video_rag"})
    db_mock.get_session = database.get_session
    cache = MagicMock()
    cache.health_check = _healthy("redis ok")
    opensearch = MagicMock()
    opensearch.health_check = _healthy("opensearch ok")
    embedder = MagicMock()
    embedder.health_check = _healthy("embedder ok")
    captioner = MagicMock()
    captioner.health_check = _healthy("caption model ok")
    return {
        "settings": settings,
        "database": db_mock,
        "cache_client": cache,
        "storage_client": storage,
        "pexels_client": MagicMock(),
        "opensearch_service": opensearch,
        "embedding_client": embedder,
        "captioner": captioner,
    }


@pytest.fixture
def client(fake_services):
    # No `with` block: the real lifespan (which connects to infra) is skipped.
    for name, value in fake_services.items():
        setattr(app.state, name, value)
    yield TestClient(app)
    app.dependency_overrides.clear()
