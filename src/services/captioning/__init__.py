from src.config import Settings
from src.services.captioning.client import Captioner, CaptionError, downscale_jpeg


def make_captioner(settings: Settings) -> Captioner:
    return Captioner(
        settings.ollama_host,
        model=settings.caption_model,
        prompt=settings.caption_prompt,
        max_side=settings.caption_max_side,
        timeout=settings.caption_timeout,
    )


__all__ = ["CaptionError", "Captioner", "downscale_jpeg", "make_captioner"]
