import io
import logging
from typing import Any

from PIL import Image

from src.services.llm import ChatModel, ModelRejected
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
    """One-sentence keyframe descriptions from a vision-language model (Ollama locally, Bedrock on AWS)."""

    def __init__(self, chat: ChatModel, prompt: str, max_side: int = 448):
        self.chat = chat
        self.prompt = prompt
        self.max_side = max_side

    @property
    def model(self) -> str:
        return self.chat.name

    def caption(self, image: bytes) -> str:
        return self.caption_with_usage(image)[0]

    def caption_with_usage(self, image: bytes) -> tuple[str, LLMCall]:
        """ModelUnavailable (retryable) propagates; a rejected request or an empty caption is a CaptionError."""
        try:
            # deterministic, one sentence
            text, call = self.chat.describe_image(self.prompt, downscale_jpeg(image, self.max_side), "caption", max_tokens=80)
        except ModelRejected as exc:
            raise CaptionError(str(exc)) from exc
        text = " ".join(text.split())
        if not text:
            raise CaptionError("empty caption")
        return text, call

    def health_check(self) -> dict[str, Any]:
        return self.chat.health()

    def close(self) -> None:
        self.chat.close()
