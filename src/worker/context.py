"""Per-process clients for Celery tasks (built on first use, after the worker forks)."""

from functools import lru_cache

from src.config import get_settings
from src.db.factory import make_database
from src.services.cache import IndexVersion, make_cache_client
from src.services.captioning import make_captioner
from src.services.embeddings import make_embedding_client
from src.services.limits import Limiter
from src.services.opensearch import make_opensearch_service
from src.services.processing.transcription import Transcriber
from src.services.processing.visual import VisualEnricher
from src.services.storage import make_storage_client
from src.services.usage import make_usage_recorder


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


@lru_cache
def get_enricher() -> VisualEnricher:
    settings = get_settings()
    limiter = Limiter(make_cache_client(settings), settings)  # captioning a user's upload counts against their tokens
    recorder = make_usage_recorder(settings, get_database(), origin="worker", limiter=limiter)
    return VisualEnricher(make_embedding_client(settings), make_captioner(settings), recorder, settings.frame_blank_max_stddev)


@lru_cache
def get_index_version() -> IndexVersion:
    """Bumped after every index write, so the API stops serving answers cached before the change."""
    return IndexVersion(make_cache_client(get_settings()))
