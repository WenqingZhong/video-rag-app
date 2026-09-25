from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=500, examples=["Give me the part where the host says ‘AI is changing everything’"]
    )
    video_id: str | None = Field(None, description="Search inside one video (e.g. the one the user uploaded)")
    kind: Literal["speech", "visual"] | None = None
    source: Literal["pexels", "upload"] | None = None
    size: int = Field(10, ge=1, le=50)
    group_by_video: bool = Field(False, description="Return only the best moment per video")


class VideoRef(BaseModel):
    id: str
    title: str | None = None
    source: str | None = None
    author: str | None = None
    source_url: str | None = None
    duration_sec: float | None = None


class SearchHitOut(BaseModel):
    video: VideoRef
    segment_id: str
    kind: str
    segment_start_sec: float
    segment_end_sec: float
    match_start_sec: float = Field(..., description="Where the match starts (exact word time for quotes)")
    match_end_sec: float
    matched_text: str | None = None
    match_score: float | None = Field(None, description="Quote alignment with the transcript, 1.0 = exact")
    score: float = Field(..., description="OpenSearch relevance score (BM25)")
    text: str | None = None
    highlight: str | None = None
    play_url: str | None = Field(None, description="Presigned video URL with #t=start,end: plays the matched moment")
    frame_url: str | None = None


class SearchResponse(BaseModel):
    query: str
    phrase: str | None = Field(None, description="Quoted phrase extracted from the query, if any")
    strategy: str = Field(..., description="exact_phrase | phrase_with_slop | fuzzy_phrase | keyword | none")
    total: int
    hits: list[SearchHitOut]


class ReindexRequest(BaseModel):
    video_id: str | None = Field(None, description="Only this video; omit to rebuild every ready video")


class JobStatus(BaseModel):
    job_id: str
    state: str
    result: Any = None
