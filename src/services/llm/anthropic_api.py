"""Claude through the Anthropic API (production: Claude Haiku 4.5).

Structured replies use tool use, as with Bedrock: one tool whose input schema is the form, and the model is
required to call it, so the reply is JSON in that shape. The API key comes from ANTHROPIC_API_KEY (SSM Parameter
Store in production, .env locally); spending can also be capped in the Anthropic console.
"""

import base64
import json
import time
from typing import Any

import anthropic

from src.services.llm.base import ModelRejected, ModelUnavailable
from src.services.usage import LLMCall, Operation

# Worth retrying later: the network, a timeout, rate limits, overload, a server error
_RETRYABLE = (
    anthropic.APIConnectionError,  # includes timeouts
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
)


class AnthropicChat:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None, timeout: float = 60.0, client: Any = None):
        self.name = model
        # The SDK retries rate limits and overload itself (twice, with backoff) before we see an error
        self.client = client or anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)

    def _create(self, **request: Any) -> Any:
        try:
            return self.client.messages.create(model=self.name, **request)
        except _RETRYABLE as exc:
            raise ModelUnavailable(f"Anthropic API {type(exc).__name__}: {exc}") from exc
        except anthropic.APIStatusError as exc:  # bad request, invalid key, no access, unknown model
            raise ModelRejected(f"Anthropic API {exc.status_code}: {exc}") from exc

    def _call(self, response: Any, operation: Operation, seconds: float, images: int = 0) -> LLMCall:
        usage = response.usage
        return LLMCall(operation, self.name, int(usage.input_tokens), int(usage.output_tokens),
                       images=images, total_sec=round(seconds, 4), output_sec=round(seconds, 4))  # fmt: skip

    def json_reply(
        self, system: str, user: str, schema: dict[str, Any], operation: Operation, max_tokens: int | None = None
    ) -> tuple[str, LLMCall]:
        started = time.perf_counter()
        response = self._create(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens or 512,
            tools=[{"name": "reply", "description": "Reply in this structure.", "input_schema": schema}],
            tool_choice={"type": "tool", "name": "reply"},
        )
        tool_input = next((block.input for block in response.content if block.type == "tool_use"), None)
        text = json.dumps(tool_input) if tool_input is not None else _text(response)
        return text, self._call(response, operation, time.perf_counter() - started)

    def describe_image(self, prompt: str, image_jpeg: bytes, operation: Operation, max_tokens: int) -> tuple[str, LLMCall]:
        started = time.perf_counter()
        image = {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(image_jpeg).decode()}
        response = self._create(
            messages=[{"role": "user", "content": [{"type": "image", "source": image}, {"type": "text", "text": prompt}]}],
            max_tokens=max_tokens,
        )
        return _text(response), self._call(response, operation, time.perf_counter() - started, images=1)

    def health(self) -> dict[str, Any]:
        # A health check that called the model would spend tokens every 30 s: report the configuration instead.
        return {"status": "healthy", "message": f"anthropic: {self.name} (checked on first call)"}

    def close(self) -> None:
        self.client.close()


def _text(response: Any) -> str:
    return "".join(block.text for block in response.content if block.type == "text")
