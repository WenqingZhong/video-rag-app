from celery.exceptions import TimeoutError as CeleryTimeout
from fastapi import APIRouter, HTTPException, status

from src.dependencies import ClipDep, Limited, SessionDep, StorageDep, ViewerDep
from src.repositories import VideoRepository
from src.schemas.api.clips import ClipRequest, ClipResponse
from src.services.clips import ClipRange

router = APIRouter(tags=["Clips"], dependencies=[Limited])


@router.post("/clips", response_model=ClipResponse)
def make_clip(body: ClipRequest, clips: ClipDep, storage: StorageDep, session: SessionDep, viewer: ViewerDep) -> ClipResponse:
    """Cut [start_sec, end_sec] of a video into an MP4 (word-accurate; cached in S3; cut by the clip worker)."""
    video = VideoRepository(session).get_visible(body.video_id, viewer)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    end = min(body.end_sec, video.duration_sec or body.end_sec)
    if end <= body.start_sec:
        raise HTTPException(status_code=422, detail="end_sec must be after start_sec and inside the video")
    clip = ClipRange(round(body.start_sec, 2), round(end, 2))
    try:
        key, cached = clips.get_or_cut(body.video_id, clip)
    except CeleryTimeout as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Clip cutting timed out") from exc
    return ClipResponse(
        url=storage.presigned_url(key),
        key=key,
        start_sec=clip.start_sec,
        end_sec=clip.end_sec,
        duration_sec=clip.duration,
        cached=cached,
    )
