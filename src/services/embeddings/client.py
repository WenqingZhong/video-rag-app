import base64
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class EmbeddingClient:
    """HTTP client for the embedding service. Vectors come back L2-normalised (cosine similarity = dot product)."""

    def __init__(self, base_url: str, timeout: float = 60.0, batch_size: int = 32):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)
        self.batch_size = batch_size
        self._model: str | None = None

    @property
    def model(self) -> str:
        """Model id reported by the service (e.g. 'ViT-B-32/laion2b_s34b_b79k'); stored with every vector."""
        if self._model is None:
            self._model = self.info()["model"]
        return self._model

    def info(self) -> dict[str, Any]:
        response = self.http.get("/health")
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict) -> list[list[float]]:
        response = self.http.post(path, json=payload)
        if response.status_code >= 500:
            response.raise_for_status()  # server problem: httpx.HTTPStatusError, treated as retryable by the worker
        if response.status_code >= 400:
            raise EmbeddingError(f"{path} rejected input: {response.text[:200]}")
        body = response.json()
        self._model = body["model"]
        return body["vectors"]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors += self._post("/embed/text", {"texts": texts[i : i + self.batch_size]})
        return vectors

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(images), self.batch_size):
            batch = [base64.b64encode(image).decode() for image in images[i : i + self.batch_size]]
            vectors += self._post("/embed/image", {"images": batch})
        return vectors

    def health_check(self) -> dict[str, Any]:
        try:
            info = self.info()
            return {"status": "healthy", "message": f"{info['model']} (dim {info['dim']})"}
        except httpx.HTTPError as exc:
            return {"status": "unhealthy", "message": f"Embedding service check failed: {exc}"}

    def close(self) -> None:
        self.http.close()
