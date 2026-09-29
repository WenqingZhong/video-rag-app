from src.config import Settings
from src.services.captioning.client import Captioner, CaptionError, downscale_jpeg
from src.services.llm import make_chat_model


def make_captioner(settings: Settings) -> Captioner:
    chat = make_chat_model(settings, "vision", timeout=settings.caption_timeout)
    return Captioner(chat, prompt=settings.caption_prompt, max_side=settings.caption_max_side)


__all__ = ["CaptionError", "Captioner", "downscale_jpeg", "make_captioner"]
