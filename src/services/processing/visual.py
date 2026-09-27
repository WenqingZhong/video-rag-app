"""Turn keyframe images into search features: a CLIP vector (meaning) and a caption (words)."""

import logging
from collections.abc import Callable

from src.services.captioning import Captioner
from src.services.embeddings import EmbeddingClient
from src.services.tracing import span
from src.services.usage import UsageRecorder

logger = logging.getLogger(__name__)


class VisualEnricher:
    """Used by the pipeline (new videos) and by video.enrich_visual (backfilling existing ones)."""

    def __init__(self, embedder: EmbeddingClient | None, captioner: Captioner | None, recorder: UsageRecorder | None = None):
        self.embedder = embedder
        self.captioner = captioner
        self.recorder = recorder  # one llm_calls row per caption, tagged with the video

    def enrich(
        self, images: list[bytes], on_stage: Callable[[str], None] = lambda stage: None, video_id: str | None = None
    ) -> list[dict]:
        fields: list[dict] = [{} for _ in images]
        if not images:
            return fields
        if self.embedder is not None:
            on_stage("embedding_frames")
            with span("stage.embedding_frames", frames=len(images)):
                vectors = self.embedder.embed_images(images)  # batched: one HTTP call per 32 frames
            for item, vector in zip(fields, vectors, strict=True):
                item.update(image_embedding=vector, embedding_model=self.embedder.model)
        if self.captioner is not None:
            on_stage("captioning")
            with span("stage.captioning", frames=len(images)) as stage:
                tokens = 0
                for i, (item, image) in enumerate(zip(fields, images, strict=True)):
                    with span("llm.caption", model=self.captioner.model, frame=i) as step:
                        caption, call = self.captioner.caption_with_usage(image)
                        step.set(prompt_tokens=call.prompt_tokens, output_tokens=call.output_tokens, prompt_sec=call.prompt_sec)
                    tokens += call.total_tokens
                    if self.recorder is not None:
                        self.recorder.record(call, "ok", video_id=video_id)
                    item.update(caption=caption, caption_model=self.captioner.model)
                    logger.info("caption %s/%s: %s", i + 1, len(images), item["caption"])
                stage.set(tokens=tokens)
        return fields
