"""Per-user limits, so a public deployment can't be flooded or run up the model bill.

Who is limited (a `Principal`): the viewer (web session or Telegram user) and, for web visitors, their IP address,
since clearing cookies starts a new anonymous session but not a new address. The bot's requests all come from one
address, so for its users only the user counts.

| limit                   | per            | counted when                                         |
|-------------------------|----------------|------------------------------------------------------|
| requests per minute     | viewer, IP     | each chat turn / search / answer request             |
| tokens per day          | viewer, IP, all| after each model call (UsageRecorder), incl. captioning the viewer's uploads |
| uploads per day         | viewer         | each upload                                          |
| Pexels downloads per day| viewer         | each download the user agrees to                     |

Counters live in Redis with a TTL (days are UTC). Tokens are checked *before* a request: the one that crosses the
line still finishes, the next is refused. If Redis is down the limits are skipped (logged): the app keeps working.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime

import redis

from src.config import Settings
from src.services.cache import CacheClient

logger = logging.getLogger(__name__)

DAY = 24 * 3600


@dataclass
class Principal:
    """Who a request (or a worker task) is for. Mutable: the viewer is known only once identity has been resolved."""

    viewer: str | None = None
    ip: str | None = None  # None for trusted services (the bot) and worker tasks
    exempt: bool = False  # admin token: evaluation scripts, Airflow


_current: ContextVar[Principal | None] = ContextVar("principal", default=None)


def current_principal() -> Principal | None:
    return _current.get()


@contextmanager
def acting_for(principal: Principal) -> Iterator[Principal]:
    token = _current.set(principal)
    try:
        yield principal
    finally:
        _current.reset(token)


class LimitExceeded(Exception):
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after


@dataclass
class Usage:
    tokens_used: int
    tokens_limit: int
    uploads_used: int
    uploads_limit: int
    pexels_used: int
    pexels_limit: int
    resets_in_sec: int

    @property
    def tokens_left(self) -> int:
        return max(self.tokens_limit - self.tokens_used, 0)


def _day() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


def seconds_to_midnight() -> int:
    return DAY - int(time.time()) % DAY


def until_reset() -> str:
    left = seconds_to_midnight()
    hours, minutes = left // 3600, left % 3600 // 60
    return f"{hours} h {minutes} min" if hours else f"{minutes} min"


class Limiter:
    def __init__(self, cache: CacheClient, settings: Settings):
        self.cache = cache
        self.s = settings

    # ---- counting --------------------------------------------------------------------------------------------
    def add_tokens(self, principal: Principal | None, tokens: int) -> None:
        """Called for every model call (UsageRecorder). Global usage counts even without a principal."""
        if tokens <= 0:
            return
        day = _day()
        keys = [f"limit:tokens:all:{day}"]
        if principal is not None and principal.viewer:
            keys.append(f"limit:tokens:viewer:{principal.viewer}:{day}")
        if principal is not None and principal.ip:
            keys.append(f"limit:tokens:ip:{principal.ip}:{day}")
        try:
            for key in keys:
                self.cache.add(key, tokens, 2 * DAY)
        except redis.RedisError as exc:
            logger.warning("token usage not counted: %s", exc)

    def _get(self, key: str) -> int:
        return int(self.cache.get(key) or 0)

    # ---- checks (raise LimitExceeded) ------------------------------------------------------------------------
    def check_request(self, principal: Principal) -> None:
        """Before anything that may call the model: the request rate, then the day's tokens."""
        if not self.s.limits_enabled or principal.exempt:
            return
        try:
            self._check_rate(principal)
            self._check_tokens(principal)
        except redis.RedisError as exc:
            logger.warning("limits not checked (Redis unavailable): %s", exc)

    def _check_rate(self, principal: Principal) -> None:
        minute = int(time.time() // 60)
        retry = 60 - int(time.time()) % 60
        checks = [(principal.viewer, self.s.limit_requests_per_min), (principal.ip, self.s.limit_ip_requests_per_min)]
        for who, limit in checks:
            if who and self.cache.add(f"limit:rate:{who}:{minute}", 1, 120) > limit:
                raise LimitExceeded(f"You're sending requests a bit fast. Please wait {retry} seconds and try again.", retry)

    def _check_tokens(self, principal: Principal) -> None:
        day, retry = _day(), seconds_to_midnight()
        if self._get(f"limit:tokens:all:{day}") >= self.s.limit_global_tokens_per_day:
            raise LimitExceeded(
                "I've reached my overall usage limit for today, so I'm taking a break. Please come back tomorrow.", retry
            )
        if principal.viewer and self._get(f"limit:tokens:viewer:{principal.viewer}:{day}") >= self.s.limit_tokens_per_day:
            raise LimitExceeded(
                f"You've used today's allowance of {self.s.limit_tokens_per_day:,} tokens. "
                f"It resets in {until_reset()} (midnight UTC).",
                retry,
            )
        if principal.ip and self._get(f"limit:tokens:ip:{principal.ip}:{day}") >= self.s.limit_ip_tokens_per_day:
            raise LimitExceeded(f"This network has used today's allowance. It resets in {until_reset()} (midnight UTC).", retry)

    def use_upload(self, principal: Principal) -> None:
        """Count one upload, or raise if the day's are used up."""
        self._use(principal, "uploads", self.s.limit_uploads_per_day,
                  f"You've uploaded {self.s.limit_uploads_per_day} videos today, the daily maximum. "
                  f"You can upload more in {until_reset()}.")  # fmt: skip

    def use_pexels(self, principal: Principal) -> None:
        self._use(principal, "pexels", self.s.limit_pexels_per_day,
                  f"I've already downloaded new videos for you {self.s.limit_pexels_per_day} times today, the daily "
                  f"maximum. You can still search everything in my library, and download more in {until_reset()}.")  # fmt: skip

    def _use(self, principal: Principal, kind: str, limit: int, message: str) -> None:
        if not self.s.limits_enabled or principal.exempt or not principal.viewer:
            return
        key = f"limit:{kind}:{principal.viewer}:{_day()}"
        try:
            if self._get(key) >= limit:
                raise LimitExceeded(message, seconds_to_midnight())
            self.cache.add(key, 1, 2 * DAY)
        except redis.RedisError as exc:
            logger.warning("%s limit not checked (Redis unavailable): %s", kind, exc)

    # ---- reporting -------------------------------------------------------------------------------------------
    def usage(self, viewer: str) -> Usage:
        day = _day()
        try:
            tokens, uploads, pexels = (self._get(f"limit:{k}:{viewer}:{day}") for k in ("tokens:viewer", "uploads", "pexels"))
        except redis.RedisError:
            tokens = uploads = pexels = 0
        return Usage(tokens, self.s.limit_tokens_per_day, uploads, self.s.limit_uploads_per_day, pexels,
                     self.s.limit_pexels_per_day, seconds_to_midnight())  # fmt: skip
