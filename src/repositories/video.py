from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from src.models import Segment, Video


def can_see(video: Video, viewer: str | None) -> bool:
    """The shared library is everyone's; an upload is only its owner's."""
    return video.owner_id is None or (viewer is not None and video.owner_id == viewer)


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
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        source: str | None = None,
        visible_to: str | None = None,
        everything: bool = False,
    ) -> tuple[list[Video], int]:
        """Videos this viewer may see: the shared library plus their own uploads (`everything`: admin and jobs only)."""
        query = select(Video)
        if not everything:
            query = query.where(
                or_(Video.owner_id.is_(None), Video.owner_id == visible_to) if visible_to else Video.owner_id.is_(None)
            )
        if status:
            query = query.where(Video.status == status)
        if source:
            query = query.where(Video.source == source)
        total = self.session.scalar(select(func.count()).select_from(query.subquery())) or 0
        items = self.session.scalars(query.order_by(Video.created_at.desc()).limit(limit).offset(offset)).all()
        return list(items), total

    def get_visible(self, video_id: str, viewer: str | None) -> Video | None:
        """The video, if this viewer may see it; otherwise None, exactly as if it didn't exist (no existence leak)."""
        video = self.get(video_id)
        return video if video is not None and can_see(video, viewer) else None

    def owned_by(self, viewer: str, limit: int = 20) -> list[Video]:
        query = select(Video).where(Video.owner_id == viewer).order_by(Video.created_at.desc()).limit(limit)
        return list(self.session.scalars(query))

    def owns_any(self, viewer: str | None) -> bool:
        if viewer is None:
            return False
        return self.session.scalar(select(Video.id).where(Video.owner_id == viewer).limit(1)) is not None

    def expired_uploads(self, older_than: datetime) -> list[Video]:
        """Owned uploads past their retention (the shared library never expires)."""
        query = select(Video).where(Video.owner_id.is_not(None), Video.created_at < older_than)
        return list(self.session.scalars(query))

    def delete_video(self, video_id: str) -> None:
        self.session.execute(delete(Segment).where(Segment.video_id == video_id))
        self.session.execute(delete(Video).where(Video.id == video_id))
        self.session.flush()

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

    def update_segment(self, segment_id: str, **fields: Any) -> None:
        segment = self.session.get(Segment, segment_id)
        if segment is not None:
            for key, value in fields.items():
                setattr(segment, key, value)
            self.session.flush()

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
