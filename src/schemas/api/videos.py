from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class VideoSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    source: str
    source_id: str | None = None
    title: str | None = None
    status: str
    stage: str | None = None
    duration_sec: float | None = None
    job_id: str | None = None
    created_at: datetime


class VideoDetail(VideoSummary):
    source_url: str | None = None
    author_name: str | None = None
    author_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    original_filename: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    has_audio: bool | None = None
    language: str | None = None
    error: str | None = None
    updated_at: datetime
    processed_at: datetime | None = None
    segment_counts: dict[str, int] = Field(default_factory=dict)
    video_url: str | None = Field(None, description="Presigned URL to stream the stored source file")


class VideoList(BaseModel):
    items: list[VideoSummary]
    total: int
    limit: int
    offset: int


class Word(BaseModel):
    word: str
    start: float
    end: float
    prob: float | None = None


class SegmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    idx: int
    start_sec: float
    end_sec: float
    text: str | None = None
    words: list[Word] | None = None
    frame_time_sec: float | None = None
    frame_url: str | None = None


class SegmentList(BaseModel):
    video_id: str
    items: list[SegmentOut]


class PexelsIngestRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100, examples=["dog"])
    count: int = Field(5, ge=1, le=50, description="How many NEW videos to queue")


class PexelsIngestResponse(BaseModel):
    query: str
    requested: int
    queued: list[VideoSummary]
    skipped_existing: int
    skipped_unsuitable: int
    pages_searched: int
