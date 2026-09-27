from src.config import Settings
from src.services.embeddings.client import EmbeddingClient, EmbeddingError


def make_embedding_client(settings: Settings) -> EmbeddingClient:
    return EmbeddingClient(settings.embedder_url, timeout=settings.embedder_timeout, batch_size=settings.embed_batch_size)


__all__ = ["EmbeddingClient", "EmbeddingError", "make_embedding_client"]
