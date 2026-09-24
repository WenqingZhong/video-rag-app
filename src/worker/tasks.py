import logging

from src.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="system.ping")
def ping() -> str:
    """Round-trip task used to verify the broker -> worker -> result backend path."""
    return "pong"
