"""Turn keyframe images into search features: a CLIP vector (meaning) and a caption (words)."""

import logging
from collections.abc import Callable

from src.services.captioning import Captioner
from src.services.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


class VisualEnricher:
    """Used by the pipeline (new videos) and by video.enrich_visual (backfilling existing ones)."""

    def __init__(self, embedder: EmbeddingClient | None, captioner: Captioner | None):
        self.embedder = embedder
        self.captioner = captioner

    def enrich(self, images: list[bytes], on_stage: Callable[[str], None] = lambda stage: None) -> list[dict]:
        fields: list[dict] = [{} for _ in images]
        if not images:
            return fields
        if self.embedder is not None:
            on_stage("embedding_frames")
            vectors = self.embedder.embed_images(images)  # batched: one HTTP call per 32 frames
            for item, vector in zip(fields, vectors, strict=True):
                item.update(image_embedding=vector, embedding_model=self.embedder.model)
        if self.captioner is not None:
            on_stage("captioning")
            for i, (item, image) in enumerate(zip(fields, images, strict=True)):
                item.update(caption=self.captioner.caption(image), caption_model=self.captioner.model)
                logger.info("caption %s/%s: %s", i + 1, len(images), item["caption"])
        return fields
