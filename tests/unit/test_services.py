from unittest.mock import MagicMock

import redis
from botocore.exceptions import ClientError

from src.config import Settings
from src.services.cache import CacheClient
from src.services.storage import StorageClient, make_storage_client
from src.worker import celery_app
from src.worker.tasks import ping


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code}}, "HeadBucket")


def test_settings_defaults_are_host_friendly():
    settings = Settings(_env_file=None)
    assert settings.redis_url.startswith("redis://localhost")
    assert settings.opensearch_host == "http://localhost:9200"


def test_cache_health_reports_redis_errors():
    raw = MagicMock()
    raw.ping.side_effect = redis.ConnectionError("refused")
    assert CacheClient(raw).health_check()["status"] == "unhealthy"


def test_ensure_bucket_creates_missing_bucket():
    s3 = MagicMock()
    s3.head_bucket.side_effect = _client_error("404")
    StorageClient(s3, "video-rag").ensure_bucket()
    s3.create_bucket.assert_called_once_with(Bucket="video-rag")


def test_ensure_bucket_propagates_auth_errors():
    s3 = MagicMock()
    s3.head_bucket.side_effect = _client_error("403")
    try:
        StorageClient(s3, "video-rag").ensure_bucket()
    except ClientError:
        pass
    else:
        raise AssertionError("403 should not be treated as a missing bucket")
    s3.create_bucket.assert_not_called()


def test_presigned_url_uses_public_endpoint():
    settings = Settings(_env_file=None, s3_endpoint_url="http://seaweedfs:8333", s3_public_endpoint_url="http://localhost:8333")
    url = make_storage_client(settings).presigned_url("clips/abc/0-5000.mp4")
    assert url.startswith("http://localhost:8333/video-rag/clips/abc/0-5000.mp4")


def test_ping_task_runs_eagerly():
    celery_app.conf.task_always_eager = True
    try:
        assert ping.delay().get() == "pong"
    finally:
        celery_app.conf.task_always_eager = False
