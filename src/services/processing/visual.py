"""Turn keyframe images into search features: a CLIP vector (meaning) and a caption (words)."""

import io
import logging
from collections.abc import Callable

from PIL import Image, ImageStat

from src.services.captioning import Captioner
from src.services.embeddings import EmbeddingClient
from src.services.tracing import span
from src.services.usage import UsageRecorder

logger = logging.getLogger(__name__)


def is_blank(image: bytes, max_stddev: float) -> bool:
    """A near-uniform frame (black screen, fade, plain background): nothing to caption or search for.

    Captioning one invents content: a plain grey frame became "a person in a dark room holding a smartphone".
    Measured on the library: blank frames 0.0, the least varied real frame (a plain sky) 13.5.
    """
    return ImageStat.Stat(Image.open(io.BytesIO(image)).convert("L")).stddev[0] <= max_stddev


class VisualEnricher:
    """Used by the pipeline (new videos) and by video.enrich_visual (backfilling existing ones)."""

    def __init__(
        self,
        embedder: EmbeddingClient | None,
        captioner: Captioner | None,
        recorder: UsageRecorder | None = None,
        blank_max_stddev: float = 4.0,
    ):
        self.embedder = embedder
        self.captioner = captioner
        self.recorder = recorder  # one llm_calls row per caption, tagged with the video
        self.blank_max_stddev = blank_max_stddev

    def enrich(
        self, images: list[bytes], on_stage: Callable[[str], None] = lambda stage: None, video_id: str | None = None
    ) -> list[dict]:
        fields: list[dict] = [{} for _ in images]
        if not images:
            return fields
        # Blank frames get neither a vector nor a caption: they stay out of visual search entirely.
        blank = [is_blank(image, self.blank_max_stddev) for image in images]
        for item, empty in zip(fields, blank, strict=True):
            if empty:
                item.update(image_embedding=None, embedding_model=None, caption=None, caption_model=None)
        todo = [i for i, empty in enumerate(blank) if not empty]
        if any(blank):
            logger.info("%s of %s keyframes are blank: not captioned or embedded", sum(blank), len(images))
        if self.embedder is not None and todo:
            on_stage("embedding_frames")
            with span("stage.embedding_frames", frames=len(todo), blank=sum(blank)):
                vectors = self.embedder.embed_images([images[i] for i in todo])  # batched: one HTTP call per 32 frames
            for i, vector in zip(todo, vectors, strict=True):
                fields[i].update(image_embedding=vector, embedding_model=self.embedder.model)
        if self.captioner is not None and todo:
            on_stage("captioning")
            with span("stage.captioning", frames=len(todo)) as stage:
                tokens = 0
                for n, i in enumerate(todo):
                    if n:  # progress for people waiting on an upload ("captioning 3/11"); the first is "captioning"
                        on_stage(f"captioning {n + 1}/{len(todo)}")
                    with span("llm.caption", model=self.captioner.model, frame=i) as step:
                        caption, call = self.captioner.caption_with_usage(images[i])
                        step.set(prompt_tokens=call.prompt_tokens, output_tokens=call.output_tokens, prompt_sec=call.prompt_sec)
                    tokens += call.total_tokens
                    if self.recorder is not None:
                        self.recorder.record(call, "ok", video_id=video_id)
                    fields[i].update(caption=caption, caption_model=self.captioner.model)
                    logger.info("caption %s/%s: %s", n + 1, len(todo), caption)
                stage.set(tokens=tokens)
        return fields
