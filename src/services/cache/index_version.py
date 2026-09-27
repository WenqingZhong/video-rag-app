"""A counter that changes whenever the search index does: answers are cached per version.

Bumped after every index write (src/services/indexing/indexer.py). A new video can turn yesterday's
"no match" into a match, so an answer cached before the change must not be served after it.
"""

import logging

import redis

from src.services.cache.client import CacheClient

logger = logging.getLogger(__name__)

KEY = "index:version"


class IndexVersion:
    def __init__(self, cache: CacheClient):
        self.cache = cache

    def current(self) -> int:
        """Raises redis.RedisError when Redis is down: the caller then skips the answer cache."""
        return int(self.cache.get(KEY) or 0)

    def bump(self) -> None:
        try:
            self.cache.incr(KEY)
        except redis.RedisError as exc:
            # The index changed but cached answers can't be invalidated: they may be stale until their TTL.
            logger.warning("index version not bumped (cached answers may be stale for up to their TTL): %s", exc)
