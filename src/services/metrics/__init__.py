"""Prometheus metrics for live request behaviour. Tokens, cost and per-step timings live in Postgres
(llm_calls, trace_spans): the workers write those too, and they need history beyond Prometheus' retention.

The API runs 4 uvicorn processes: with PROMETHEUS_MULTIPROC_DIR set, each writes its samples to that
directory and /metrics adds them up (prometheus_client's multiprocess mode).
"""

import os

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, CollectorRegistry, Counter, Histogram, generate_latest, multiprocess

_SECONDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30, 60)

HTTP_REQUESTS = Counter("http_requests_total", "API requests", ["method", "route", "status"])
HTTP_SECONDS = Histogram("http_request_duration_seconds", "API request duration", ["method", "route"], buckets=_SECONDS)
ASK_REQUESTS = Counter(
    "ask_requests_total",
    "/ask requests by result, by what understood the request, and by cache result",
    ["status", "understood_by", "answer_cache", "understanding_cache"],
)
ASK_STAGE_SECONDS = Histogram("ask_stage_seconds", "/ask time per stage", ["stage"], buckets=_SECONDS)
CHAT_TURNS = Counter("chat_turns_total", "Chat turns by the action taken and what decided it", ["action", "decided_by"])


def observe_request(method: str, route: str, status: int, seconds: float) -> None:
    HTTP_REQUESTS.labels(method, route, str(status)).inc()
    HTTP_SECONDS.labels(method, route).observe(seconds)


def observe_ask(status: str, understood_by: str, cache: dict[str, str], timings: dict[str, float]) -> None:
    ASK_REQUESTS.labels(status, understood_by, cache.get("answer", "off"), cache.get("understanding", "off")).inc()
    for stage, seconds in timings.items():
        if stage != "total":
            ASK_STAGE_SECONDS.labels(stage).observe(seconds)


def observe_chat(action: str, decided_by: str) -> None:
    CHAT_TURNS.labels(action, decided_by).inc()


def render() -> tuple[bytes, str]:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
