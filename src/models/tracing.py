from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.interfaces.postgresql import Base


class SpanRecord(Base):
    """One timed step of a request or task. Spans sharing a trace_id form one trace (a tree via parent_id)."""

    __tablename__ = "trace_spans"

    span_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(String(16))  # null: the root of the trace
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    service: Mapped[str] = mapped_column(String(16), nullable=False)  # "api" | "worker" | "clip-worker"
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # "ok" | "error"
    error: Mapped[str | None] = mapped_column(Text)
    attributes: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_trace_spans_trace_id", "trace_id"),
        Index("ix_trace_spans_name_started_at", "name", "started_at"),
        Index("ix_trace_spans_started_at", "started_at"),
    )
