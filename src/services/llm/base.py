"""The one interface every model call goes through, so where the model runs is a setting:
local development → Ollama (a container); production → Claude through the Anthropic API, or Amazon Bedrock, or a
GPU server running Ollama.

Two operations cover everything the app asks a model:
  json_reply      fill a form given as a JSON schema (request understanding, question answering, chat routing)
  describe_image  one sentence about a keyframe (captions)
Both return the text and an LLMCall (tokens, time) for the usage ledger.
"""

from typing import Any, Protocol

from src.services.usage import LLMCall, Operation


class ModelError(RuntimeError):
    """Any failure to get a reply from the model."""


class ModelUnavailable(ModelError):
    """Down, timed out, overloaded or throttled: worth retrying later."""


class ModelRejected(ModelError):
    """The request itself was refused (bad input, no access to the model): retrying won't help."""


class ChatModel(Protocol):
    provider: str  # "ollama" | "anthropic" | "bedrock"
    name: str  # the model id, recorded with every call

    def json_reply(
        self, system: str, user: str, schema: dict[str, Any], operation: Operation, max_tokens: int | None = None
    ) -> tuple[str, LLMCall]:
        """The model's reply as JSON text matching `schema` (parsing and validation stay with the caller)."""
        ...

    def describe_image(self, prompt: str, image_jpeg: bytes, operation: Operation, max_tokens: int) -> tuple[str, LLMCall]: ...

    def health(self) -> dict[str, Any]: ...

    def close(self) -> None: ...
