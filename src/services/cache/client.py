import logging
from typing import Any

import redis

logger = logging.getLogger(__name__)


class CacheClient:
    """Thin wrapper around Redis used for response caching and short-lived state."""

    def __init__(self, client: redis.Redis):
        self.client = client

    def get(self, key: str) -> str | None:
        return self.client.get(key)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        self.client.set(key, value, ex=ttl_seconds)

    def incr(self, key: str) -> int:
        return int(self.client.incr(key))

    def health_check(self) -> dict[str, Any]:
        try:
            self.client.ping()
            return {"status": "healthy", "message": "Connected successfully"}
        except redis.RedisError as exc:
            return {"status": "unhealthy", "message": f"Redis ping failed: {exc}"}

    def close(self) -> None:
        self.client.close()
