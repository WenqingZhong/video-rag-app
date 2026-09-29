"""Models served by Ollama (local development: the `ollama` container)."""

import base64
from typing import Any

import httpx

from src.services.llm.base import ModelRejected, ModelUnavailable
from src.services.usage import LLMCall, Operation


class OllamaChat:
    provider = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 60.0):
        self.http = httpx.Client(base_url=host, timeout=timeout)
        self.name = model

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.http.post(path, json=payload)
        except httpx.HTTPError as exc:  # connection refused, timeout
            raise ModelUnavailable(f"Ollama unreachable: {exc}") from exc
        if response.status_code >= 500:
            raise ModelUnavailable(f"Ollama {response.status_code}: {response.text[:200]}")
        if response.status_code >= 400:
            raise ModelRejected(f"Ollama rejected the request: {response.text[:200]}")
        return response.json()

    def json_reply(
        self, system: str, user: str, schema: dict[str, Any], operation: Operation, max_tokens: int | None = None
    ) -> tuple[str, LLMCall]:
        options: dict[str, Any] = {"temperature": 0}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body = self._post(
            "/api/chat",
            {
                "model": self.name,
                "stream": False,
                "format": schema,  # Ollama constrains the reply to this JSON schema
                "options": options,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        return (body.get("message") or {}).get("content", ""), LLMCall.from_ollama(body, operation, self.name)

    def describe_image(self, prompt: str, image_jpeg: bytes, operation: Operation, max_tokens: int) -> tuple[str, LLMCall]:
        body = self._post(
            "/api/generate",
            {
                "model": self.name,
                "prompt": prompt,
                "images": [base64.b64encode(image_jpeg).decode()],
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
        )
        return body.get("response", ""), LLMCall.from_ollama(body, operation, self.name, images=1)

    def health(self) -> dict[str, Any]:
        try:
            response = self.http.get("/api/tags", timeout=5)
            response.raise_for_status()
            models = {m["name"] for m in response.json().get("models", [])}
        except httpx.HTTPError as exc:
            return {"status": "unhealthy", "message": f"Ollama check failed: {exc}"}
        if self.name not in models:
            return {"status": "unhealthy", "message": f"model '{self.name}' not pulled (ollama pull {self.name})"}
        return {"status": "healthy", "message": f"ollama: {self.name} available"}

    def close(self) -> None:
        self.http.close()
