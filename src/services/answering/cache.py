"""Remember whole /ask answers: a repeated request skips the model, the search and the clip lookup.

Key: the request + filters + the index version + a fingerprint of the code and settings behind an answer.
Every index write bumps the version, so a video added since can't be hidden behind an old "no match".
Clip files live in S3; only their keys are cached (presigned URLs are made fresh for every response).
"""

import json
import logging
from dataclasses import asdict

import redis

from src.config import Settings
from src.services.cache import CacheClient, IndexVersion, code_fingerprint, digest, normalize_request
from src.services.clips import ClipRange
from src.services.search import SearchHit
from src.services.understanding.cache import understanding_fields, understanding_to_dict

logger = logging.getLogger(__name__)

# The code an answer depends on. Settings: thresholds, clip lengths, anything that changes results.
_PACKAGES = ("services/understanding", "services/search", "services/answering", "services/clips")
# Includes where the model runs: answers from one model are never served for another
_SETTING_PREFIXES = ("search_", "clip_", "understanding_model", "opensearch_index", "llm_provider", "bedrock_text_model")


class AnswerCache:
    def __init__(self, cache: CacheClient, versions: IndexVersion, settings: Settings):
        self.cache = cache
        self.versions = versions
        self.ttl = settings.answer_cache_ttl_sec
        relevant = {k: v for k, v in settings.model_dump().items() if k.startswith(_SETTING_PREFIXES)}
        self.prefix = f"answer:{code_fingerprint(*_PACKAGES, settings=relevant)}"

    def key(self, query: str, video_id: str | None, source: str | None, max_clips: int, scope: str | None = None) -> str | None:
        """None when Redis is down: the request is answered without the cache."""
        try:
            version = self.versions.current()
        except redis.RedisError as exc:
            logger.warning("answer cache unavailable: %s", exc)
            return None
        # scope: a viewer with their own uploads gets their own entries; everyone else shares the library's
        return f"{self.prefix}:v{version}:{digest(normalize_request(query), video_id, source, max_clips, scope)}"

    def get(self, key: str) -> dict | None:
        try:
            raw = self.cache.get(key)
        except redis.RedisError as exc:
            logger.warning("answer cache unavailable: %s", exc)
            return None
        if raw is None:
            return None
        data = json.loads(raw)
        data["understanding"] = understanding_fields(data["understanding"])
        data["clips"] = [
            {**c, "hit": SearchHit(**c["hit"]), "clip": ClipRange(c["clip"]["start_sec"], c["clip"]["end_sec"])}
            for c in data["clips"]
        ]
        return data

    def put(self, key: str, answer) -> None:
        data = {
            "understanding": understanding_to_dict(answer.understanding),
            "status": answer.status,
            "strategy": answer.strategy,
            "answer": answer.answer,
            "clips": [
                {
                    "hit": asdict(c.hit),
                    "clip": {"start_sec": c.clip.start_sec, "end_sec": c.clip.end_sec},
                    "key": c.key,
                    "explanation": c.explanation,
                }
                for c in answer.clips
            ],
        }
        try:
            self.cache.set(key, json.dumps(data, default=str), ttl_seconds=self.ttl)
        except redis.RedisError as exc:
            logger.warning("answer not cached: %s", exc)
