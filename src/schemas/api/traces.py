from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CleanupRequest(BaseModel):
    older_than_days: int | None = Field(None, ge=1, description="Default: TRACE_RETENTION_DAYS")


class CleanupResponse(BaseModel):
    deleted_spans: int
    before: datetime


class SpanOut(BaseModel):
    span_id: str
    parent_id: str | None
    name: str
    service: str
    started_at: datetime
    duration_ms: float
    status: str
    error: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TraceSummary(SpanOut):
    trace_id: str


class TraceLLM(BaseModel):
    calls: int
    tokens: int
    cost_usd: float


class TraceOut(BaseModel):
    trace_id: str
    spans: list[SpanOut]
    llm: TraceLLM = Field(..., description="Model calls made for this trace (from llm_calls)")
    waterfall: str = Field(..., description="Text timeline: one line per span, children indented")


class UploadCleanupResponse(BaseModel):
    deleted: int
    video_ids: list[str]
    retention_days: int
