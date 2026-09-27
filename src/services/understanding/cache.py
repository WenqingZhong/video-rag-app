"""Remember how a request was understood, so the same request never pays for the model twice.

Key: the request (lowercased, spacing collapsed) + the model + a fingerprint of this package's code, which
includes the prompt and the guards. The model runs at temperature 0, so asking again would give the same answer.
"""

import json
import logging

import redis

from src.services.cache import CacheClient, code_fingerprint, digest, normalize_request
from src.services.understanding.intent import Intent
from src.services.usage import LLMCall

logger = logging.getLogger(__name__)


def understanding_to_dict(result) -> dict:
    call = result.llm_call or result.avoided_call  # the model call behind this result, if any
    return {
        "intent": result.intent.model_dump(),
        "source": result.source,
        "notes": result.notes,
        "llm_raw": result.llm_raw.model_dump() if result.llm_raw else None,
        "llm_outcome": result.llm_outcome,
        # the call a later hit avoids: what the dashboard counts as saved
        "llm": {"model": call.model, "prompt_tokens": call.prompt_tokens, "output_tokens": call.output_tokens} if call else None,
    }


def understanding_fields(data: dict) -> dict:
    """Keyword arguments for Understanding(...), plus `avoided`: the model call a cache hit replaces (or None)."""
    llm = data["llm"]
    return {
        "intent": Intent(**data["intent"]),
        "source": data["source"],
        "notes": data["notes"],
        "llm_raw": Intent(**data["llm_raw"]) if data["llm_raw"] else None,
        "llm_outcome": data["llm_outcome"],
        "avoided": LLMCall("understand", llm["model"], llm["prompt_tokens"], llm["output_tokens"]) if llm else None,
    }


class UnderstandingCache:
    def __init__(self, cache: CacheClient, model: str, ttl_seconds: int):
        self.cache = cache
        self.ttl = ttl_seconds
        self.prefix = f"understand:{code_fingerprint('services/understanding', settings={'model': model})}"

    def key(self, query: str) -> str:
        return f"{self.prefix}:{digest(normalize_request(query))}"

    def get(self, query: str) -> dict | None:
        """The stored result, with `avoided` = the model call it replaces. None on a miss or if Redis is down."""
        try:
            raw = self.cache.get(self.key(query))
        except redis.RedisError as exc:
            logger.warning("understanding cache unavailable: %s", exc)
            return None
        if raw is None:
            return None
        return understanding_fields(json.loads(raw))

    def put(self, query: str, result) -> None:
        try:
            self.cache.set(self.key(query), json.dumps(understanding_to_dict(result)), ttl_seconds=self.ttl)
        except redis.RedisError as exc:
            logger.warning("understanding not cached: %s", exc)
