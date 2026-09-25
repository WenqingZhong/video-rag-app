from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from src.models import Segment, Video


class VideoRepository:
    """Data access for videos and their segments. Callers own the transaction (commit/rollback)."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, **fields: Any) -> Video:
        video = Video(**fields)
        self.session.add(video)
        self.session.flush()
        return video

    def get(self, video_id: str) -> Video | None:
        return self.session.get(Video, video_id)

    def get_by_source(self, source: str, source_id: str) -> Video | None:
        return self.session.scalar(select(Video).where(Video.source == source, Video.source_id == source_id))

    def existing_source_ids(self, source: str, source_ids: list[str]) -> set[str]:
        if not source_ids:
            return set()
        rows = self.session.scalars(select(Video.source_id).where(Video.source == source, Video.source_id.in_(source_ids)))
        return set(rows)

    def list_videos(
        self, limit: int = 50, offset: int = 0, status: str | None = None, source: str | None = None
    ) -> tuple[list[Video], int]:
        query = select(Video)
        if status:
            query = query.where(Video.status == status)
        if source:
            query = query.where(Video.source == source)
        total = self.session.scalar(select(func.count()).select_from(query.subquery())) or 0
        items = self.session.scalars(query.order_by(Video.created_at.desc()).limit(limit).offset(offset)).all()
        return list(items), total

    def update(self, video_id: str, **fields: Any) -> Video | None:
        video = self.get(video_id)
        if video is None:
            return None
        for key, value in fields.items():
            setattr(video, key, value)
        self.session.flush()
        return video

    def set_status(self, video_id: str, status: str, stage: str | None = None, error: str | None = None) -> None:
        fields: dict[str, Any] = {"status": status, "stage": stage, "error": error}
        if status == "ready":
            fields["processed_at"] = datetime.now(UTC)
        self.update(video_id, **fields)

    def replace_segments(self, video_id: str, segments: list[dict[str, Any]]) -> int:
        """Delete then insert, so re-processing a video (retries, re-runs) never duplicates segments."""
        self.session.execute(delete(Segment).where(Segment.video_id == video_id))
        self.session.add_all(Segment(video_id=video_id, **fields) for fields in segments)
        self.session.flush()
        return len(segments)

    def list_segments(self, video_id: str, kind: str | None = None) -> list[Segment]:
        query = select(Segment).where(Segment.video_id == video_id)
        if kind:
            query = query.where(Segment.kind == kind)
        return list(self.session.scalars(query.order_by(Segment.kind, Segment.idx)))

    def segment_counts(self, video_id: str) -> dict[str, int]:
        rows = self.session.execute(
            select(Segment.kind, func.count()).where(Segment.video_id == video_id).group_by(Segment.kind)
        ).all()
        return {kind: count for kind, count in rows}
