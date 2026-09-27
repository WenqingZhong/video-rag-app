from pydantic import BaseModel, Field


class ClipRequest(BaseModel):
    video_id: str
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., gt=0)


class ClipResponse(BaseModel):
    url: str = Field(..., description="Presigned URL of the cut MP4")
    key: str = Field(..., description="S3 key (clips/{video_id}/{start_ms}-{end_ms}.mp4)")
    start_sec: float
    end_sec: float
    duration_sec: float
    cached: bool = Field(..., description="True if this exact clip had been cut before (served from S3)")
