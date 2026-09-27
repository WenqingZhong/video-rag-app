from datetime import UTC, datetime

from sqlalchemy import BigInteger, DateTime, Float, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.interfaces.postgresql import Base


def _now() -> datetime:
    return datetime.now(UTC)


class LLMCallRecord(Base):
    """One model call that consumed tokens. The ledger behind the token and cost dashboards.

    No foreign key to videos: usage history must survive a deleted video.
    """

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, nullable=False)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)  # "understand" | "caption"
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)  # see src/services/usage/models.py
    origin: Mapped[str] = mapped_column(String(16), nullable=False)  # "api" | "worker" | "eval"
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)  # includes image tokens
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    load_sec: Mapped[float] = mapped_column(Float, nullable=False)  # loading the model into memory
    prompt_sec: Mapped[float] = mapped_column(Float, nullable=False)  # reading the prompt (low when Ollama reuses it)
    output_sec: Mapped[float] = mapped_column(Float, nullable=False)  # generating the reply
    total_sec: Mapped[float] = mapped_column(Float, nullable=False)
    # Priced when the call is made, like a bill: a later price change does not rewrite history.
    cost_usd: Mapped[float] = mapped_column(Numeric(14, 8), nullable=False)
    price_reference: Mapped[str] = mapped_column(String(64), nullable=False)
    # outcome "cache_hit": no tokens spent; what the avoided call would have cost (Week 6, Phase 3)
    saved_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    saved_cost_usd: Mapped[float] = mapped_column(Numeric(14, 8), nullable=False, default=0, server_default="0")
    video_id: Mapped[str | None] = mapped_column(String(36))
    request_id: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        Index("ix_llm_calls_created_at", "created_at"),
        Index("ix_llm_calls_video_id", "video_id"),
        Index("ix_llm_calls_request_id", "request_id"),
    )
