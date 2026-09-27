import base64
import io
import logging
from typing import Any

import httpx
from PIL import Image

from src.services.usage import LLMCall

logger = logging.getLogger(__name__)


class CaptionError(RuntimeError):
    pass


def downscale_jpeg(image: bytes, max_side: int) -> bytes:
    """Vision models cost scale with pixel count: shrink to max_side on the long edge before sending."""
    picture = Image.open(io.BytesIO(image))
    picture.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    picture.convert("RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class Captioner:
    """One-sentence keyframe descriptions from a vision-language model served by Ollama."""

    def __init__(self, ollama_host: str, model: str, prompt: str, max_side: int = 448, timeout: float = 180.0):
        self.http = httpx.Client(base_url=ollama_host, timeout=timeout)
        self.model = model
        self.prompt = prompt
        self.max_side = max_side

    def caption(self, image: bytes) -> str:
        return self.caption_with_usage(image)[0]

    def caption_with_usage(self, image: bytes) -> tuple[str, LLMCall]:
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [base64.b64encode(downscale_jpeg(image, self.max_side)).decode()],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 80},  # deterministic, one sentence
        }
        response = self.http.post("/api/generate", json=payload)
        if response.status_code >= 500:
            response.raise_for_status()  # retryable
        if response.status_code >= 400:
            raise CaptionError(f"Ollama rejected the request: {response.text[:200]}")
        body = response.json()
        text = " ".join(body.get("response", "").split())
        if not text:
            raise CaptionError("empty caption")
        return text, LLMCall.from_ollama(body, "caption", self.model, images=1)

    def health_check(self) -> dict[str, Any]:
        try:
            response = self.http.get("/api/tags", timeout=5)
            response.raise_for_status()
            models = {m["name"] for m in response.json().get("models", [])}
        except httpx.HTTPError as exc:
            return {"status": "unhealthy", "message": f"Ollama check failed: {exc}"}
        if self.model not in models:
            return {"status": "unhealthy", "message": f"caption model '{self.model}' not pulled (ollama pull {self.model})"}
        return {"status": "healthy", "message": f"caption model {self.model} available"}

    def close(self) -> None:
        self.http.close()
