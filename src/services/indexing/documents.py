"""Postgres rows → OpenSearch documents. Pure function: the index is a derived copy of Postgres."""

from datetime import UTC, datetime
from typing import Any

from src.models import Segment, Video


def segment_documents(video: Video, segments: list[Segment]) -> list[dict[str, Any]]:
    now = datetime.now(UTC).isoformat()
    video_fields = {
        "video_id": video.id,
        # Visual segments have no text until Week 4 captions; the title makes them findable by keyword meanwhile.
        "video_title": video.title,
        "video_source": video.source,
        "video_source_id": video.source_id,
        "video_author": video.author_name,
        "video_source_url": video.source_url,
        "video_s3_key": video.s3_key,
        "video_duration_sec": video.duration_sec,
        "language": video.language,
        "indexed_at": now,
    }
    return [
        {
            "segment_id": s.id,
            "kind": s.kind,
            "idx": s.idx,
            "start_sec": s.start_sec,
            "end_sec": s.end_sec,
            "text": s.text,
            "words": s.words,
            "frame_key": s.frame_key,
            "frame_time_sec": s.frame_time_sec,
            "caption": s.caption,
            "image_embedding": s.image_embedding,
            "embedding_model": s.embedding_model,
            "caption_model": s.caption_model,
            **video_fields,
        }
        for s in segments
    ]
