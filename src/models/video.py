import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.interfaces.postgresql import Base


def _now() -> datetime:
    return datetime.now(UTC)


class VideoSource(enum.StrEnum):
    PEXELS = "pexels"
    UPLOAD = "upload"


class VideoStatus(enum.StrEnum):
    QUEUED = "queued"  # row exists, waiting for a worker
    DOWNLOADING = "downloading"  # Pexels only: fetching the file into S3
    PROCESSING = "processing"  # worker is running the pipeline (see `stage`)
    READY = "ready"  # segments stored; searchable once indexed (Week 3)
    FAILED = "failed"  # see `error`


class SegmentKind(enum.StrEnum):
    SPEECH = "speech"  # transcript window with word-level timestamps
    VISUAL = "visual"  # a shot (or part of a long shot) with a keyframe in S3


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    source_id: Mapped[str | None] = mapped_column(String(64))  # Pexels video id; null for uploads
    title: Mapped[str | None] = mapped_column(String(512))
    source_url: Mapped[str | None] = mapped_column(Text)  # Pexels page (attribution)
    source_file_url: Mapped[str | None] = mapped_column(Text)  # Pexels rendition to download
    author_name: Mapped[str | None] = mapped_column(String(256))
    author_url: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    original_filename: Mapped[str | None] = mapped_column(String(512))
    content_type: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    s3_key: Mapped[str | None] = mapped_column(Text)

    # Filled in by the pipeline (ffprobe / whisper)
    duration_sec: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    has_audio: Mapped[bool | None] = mapped_column(Boolean)
    language: Mapped[str | None] = mapped_column(String(16))

    status: Mapped[str] = mapped_column(String(16), nullable=False, default=VideoStatus.QUEUED)
    stage: Mapped[str | None] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(Text)
    job_id: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now, nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    segments: Mapped[list["Segment"]] = relationship(
        back_populates="video", cascade="all, delete-orphan", passive_deletes=True, order_by="Segment.start_sec"
    )

    __table_args__ = (
        # One row per Pexels video: makes re-running ingestion idempotent.
        UniqueConstraint("source", "source_id", name="uq_videos_source_source_id"),
        Index("ix_videos_status", "status"),
        Index("ix_videos_created_at", "created_at"),
    )


class Segment(Base):
    """A time range of a video that search will return. Start/end are seconds from the video start."""

    __tablename__ = "segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)  # order within (video, kind)
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str | None] = mapped_column(Text)  # speech: transcript of the window
    words: Mapped[list | None] = mapped_column(JSON)  # speech: [{"word", "start", "end", "prob"}]
    frame_key: Mapped[str | None] = mapped_column(Text)  # visual: S3 key of the keyframe
    frame_time_sec: Mapped[float | None] = mapped_column(Float)  # visual: where the keyframe was taken
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)

    video: Mapped[Video] = relationship(back_populates="segments")

    __table_args__ = (
        UniqueConstraint("video_id", "kind", "idx", name="uq_segments_video_kind_idx"),
        Index("ix_segments_video_id_kind", "video_id", "kind"),
    )
