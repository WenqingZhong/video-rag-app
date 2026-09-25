"""Per-process clients for Celery tasks (built on first use, after the worker forks)."""

from functools import lru_cache

from src.config import get_settings
from src.db.factory import make_database
from src.services.opensearch import make_opensearch_service
from src.services.processing.transcription import Transcriber
from src.services.storage import make_storage_client


@lru_cache
def get_database():
    return make_database()


@lru_cache
def get_storage():
    storage = make_storage_client(get_settings())
    storage.ensure_bucket()
    return storage


@lru_cache
def get_transcriber() -> Transcriber:
    s = get_settings()
    return Transcriber(
        s.whisper_model, device=s.whisper_device, compute_type=s.whisper_compute_type, beam_size=s.whisper_beam_size
    )


@lru_cache
def get_opensearch():
    service = make_opensearch_service(get_settings())
    service.ensure_index()
    return service
