from pydantic import BaseModel, Field


class ChatClip(BaseModel):
    url: str
    explanation: str
    key: str | None = Field(None, description="S3 key of the clip file (for clients that fetch it from storage)")
    video_id: str | None = None
    title: str | None = None
    source: str | None = None
    author: str | None = None
    source_url: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None


class ChatCitation(BaseModel):
    video_id: str
    title: str
    kind: str
    start_sec: float
    end_sec: float
    text: str


class ChatUpdateResponse(BaseModel):
    conversation_id: str
    status: str = Field(..., description="idle (nothing downloading) | pending | done | timed_out")
    reply: str | None = Field(None, description="done / timed_out: the message to show")
    clips: list[ChatClip] = Field(default_factory=list)
    progress: dict[str, int] = Field(default_factory=dict, description="Downloaded videos: ready / pending / failed")
    status_text: str | None = Field(None, description="While your upload processes: which step it's on, and for how long")


class ChatResponse(BaseModel):
    conversation_id: str = Field(..., description="Send it back with the next message to continue the conversation")
    request_id: str | None = None
    reply: str
    action: str = Field(..., description="find_clip | find_by_image | answer_question | fetch_from_pexels | list_videos | reply")
    decided_by: str = Field(..., description="rules (a clear pattern) | llm (the model, checked by guards)")
    notes: list[str] = Field(default_factory=list, description="Why the model's choice was changed, if it was")
    clips: list[ChatClip] = Field(default_factory=list)
    citations: list[ChatCitation] = Field(default_factory=list)
    videos: list[dict] = Field(default_factory=list)
    fetching: list[str] = Field(default_factory=list, description="Video ids being downloaded for this conversation")
    uploading: str | None = Field(None, description="Your uploaded video's id while it's processed: poll /updates")
    seconds: float
