from collections.abc import Callable
from typing import Any

import httpx
from fastapi import APIRouter

from src.config import Settings
from src.dependencies import CacheDep, DatabaseDep, OpenSearchDep, SettingsDep, StorageDep
from src.schemas.api.health import HealthResponse, ServiceStatus
from src.worker.celery_app import celery_app

router = APIRouter(tags=["Health"])


@router.get("/ping")
def ping() -> dict:
    """Liveness probe: the process is up. Does not touch dependencies."""
    return {"status": "ok", "message": "pong"}


def _check_http(url: str, name: str) -> dict[str, Any]:
    try:
        response = httpx.get(url, timeout=3.0)
        response.raise_for_status()
        return {"status": "healthy", "message": f"{name} reachable"}
    except httpx.HTTPError as exc:
        return {"status": "unhealthy", "message": f"{name} check failed: {exc}"}


def _check_worker() -> dict[str, Any]:
    try:
        # limit=1: return as soon as one worker answers instead of always waiting the full timeout
        replies = celery_app.control.ping(timeout=1.0, limit=1)
    except Exception as exc:  # noqa: BLE001 - kombu/redis raise several unrelated types
        return {"status": "unhealthy", "message": f"Broker unreachable: {exc}"}
    if not replies:
        return {"status": "unhealthy", "message": "No Celery workers responded"}
    return {"status": "healthy", "message": "Celery worker responding"}


def _checks(settings: Settings, database, cache, storage, opensearch) -> dict[str, Callable[[], dict[str, Any]]]:
    return {
        "database": database.healthcheck,
        "redis": cache.health_check,
        "object_storage": storage.health_check,
        "opensearch": opensearch.health_check,
        "ollama": lambda: _check_http(f"{settings.ollama_host}/api/version", "Ollama"),
        "worker": _check_worker,
    }


@router.get("/health", response_model=HealthResponse)
def health_check(
    settings: SettingsDep, database: DatabaseDep, cache: CacheDep, storage: StorageDep, opensearch: OpenSearchDep
) -> HealthResponse:
    """Readiness probe: reports every dependency. Returns 200 with `degraded` if any is down."""
    services: dict[str, ServiceStatus] = {}
    for name, check in _checks(settings, database, cache, storage, opensearch).items():
        result = check()
        status = "healthy" if result.get("status") == "healthy" else "unhealthy"
        message = (
            result.get("message") or result.get("error") or ("Connected successfully" if status == "healthy" else "Unavailable")
        )
        services[name] = ServiceStatus(status=status, message=message)

    overall = "ok" if all(s.status == "healthy" for s in services.values()) else "degraded"
    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
