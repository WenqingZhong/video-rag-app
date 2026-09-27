from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.api.search import VideoRef


class AskRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=500, examples=["Give me the part where the host says AI is changing everything"]
    )
    video_id: str | None = Field(None, description="Look only inside this video (e.g. the user's upload)")
    source: Literal["pexels", "upload"] | None = None
    max_clips: int = Field(1, ge=1, le=3)


class IntentOut(BaseModel):
    type: Literal["quote", "topic", "visual"]
    phrase: str | None = None
    topic: str | None = None
    visual: str | None = None
    exclude: list[str] = Field(default_factory=list)


class Understood(BaseModel):
    intent: IntentOut
    source: str = Field(..., description="llm | llm+rules | rules")
    notes: list[str] = Field(default_factory=list, description="What the guards changed, and why")


class ClipOut(BaseModel):
    url: str = Field(..., description="Presigned URL of the cut MP4")
    start_sec: float
    end_sec: float
    duration_sec: float
    cached: bool
    video: VideoRef
    kind: str
    matched_text: str | None = None
    match_score: float | None = None
    caption: str | None = None
    explanation: str


class AskResponse(BaseModel):
    query: str
    status: Literal["answered", "no_match", "needs_subject"]
    answer: str = Field(..., description="Template text: the best clip, why nothing matched, or a question back")
    understood: Understood
    strategy: str
    clips: list[ClipOut]
    timings: dict[str, float] = Field(..., description="Seconds per stage: understand, search, clips, total")
