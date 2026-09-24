import redis

from src.config import Settings
from src.services.cache.client import CacheClient


def make_cache_client(settings: Settings) -> CacheClient:
    """Create a Redis-backed cache client. Connections are lazy, so this never blocks startup."""
    client = redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=settings.redis_socket_timeout,
        socket_connect_timeout=settings.redis_socket_timeout,
        health_check_interval=30,
    )
    return CacheClient(client)
