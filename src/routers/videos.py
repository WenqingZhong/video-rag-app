from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status

from src.dependencies import (
    AdminOnly,
    DeletionDep,
    IngestionDep,
    LimiterDep,
    PexelsDep,
    SessionDep,
    SettingsDep,
    StorageDep,
    ViewerDep,
    principal_for,
    too_many,
)
from src.models import VideoStatus
from src.repositories import VideoRepository
from src.schemas.api.videos import (
    PexelsIngestRequest,
    PexelsIngestResponse,
    SegmentList,
    SegmentOut,
    VideoDetail,
    VideoList,
    VideoSummary,
)
from src.services.identity import is_admin
from src.services.ingestion import VideoBusy
from src.services.limits import LimitExceeded
from src.services.pexels import PexelsError

router = APIRouter(prefix="/videos", tags=["Videos"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


@router.post("", response_model=VideoSummary, status_code=status.HTTP_202_ACCEPTED)
def upload_video(
    request: Request,
    ingestion: IngestionDep,
    settings: SettingsDep,
    viewer: ViewerDep,
    limiter: LimiterDep,
    file: Annotated[UploadFile, File()],
) -> VideoSummary:
    """Store the file in S3, create the row, queue processing. Returns immediately (202); the worker does the rest.
    The video is private to the uploader and deleted after UPLOAD_RETENTION_DAYS. With the admin token it joins the
    shared library instead (no owner, never expires)."""
    extension = Path(file.filename or "").suffix.lower()
    is_video_type = (file.content_type or "").startswith("video/")
    if extension not in ALLOWED_EXTENSIONS and not is_video_type:
        raise HTTPException(status_code=415, detail=f"Unsupported file type; expected one of {sorted(ALLOWED_EXTENSIONS)}")
    max_bytes = settings.upload_max_mb * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(status_code=413, detail=f"File larger than {settings.upload_max_mb} MB")

    admin = is_admin(request, settings)
    if limiter is not None and not admin:
        try:
            limiter.use_upload(principal_for(viewer))
        except LimitExceeded as exc:
            raise too_many(exc) from exc
    owner = None if admin else viewer
    video = ingestion.create_upload(file.file, file.filename, file.content_type, file.size, owner_id=owner)
    return VideoSummary.model_validate(video)


@router.post("/pexels", response_model=PexelsIngestResponse, status_code=status.HTTP_202_ACCEPTED, dependencies=[AdminOnly])
def ingest_pexels(body: PexelsIngestRequest, ingestion: IngestionDep, pexels: PexelsDep) -> PexelsIngestResponse:
    """Search Pexels and queue up to `count` new videos (already-ingested ones are skipped)."""
    try:
        result = ingestion.ingest_pexels(pexels, body.query, body.count)
    except PexelsError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return PexelsIngestResponse(
        query=result.query,
        requested=result.requested,
        queued=[VideoSummary.model_validate(v) for v in result.queued],
        skipped_existing=result.skipped_existing,
        skipped_unsuitable=result.skipped_unsuitable,
        pages_searched=result.pages_searched,
    )


@router.post("/{video_id}/reprocess", response_model=VideoSummary, status_code=status.HTTP_202_ACCEPTED, dependencies=[AdminOnly])
def reprocess_video(video_id: str, ingestion: IngestionDep, viewer: ViewerDep) -> VideoSummary:
    """Re-run the pipeline (e.g. after `failed`, or after changing processing settings). Idempotent."""
    video = ingestion.repo.get_visible(video_id, viewer)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status in (VideoStatus.QUEUED, VideoStatus.DOWNLOADING, VideoStatus.PROCESSING):
        raise HTTPException(status_code=409, detail=f"Video is already {video.status}")
    return VideoSummary.model_validate(ingestion.reprocess(video))


@router.get("", response_model=VideoList)
def list_videos(
    session: SessionDep,
    viewer: ViewerDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status_filter: str | None = Query(None, alias="status"),
    source: str | None = None,
) -> VideoList:
    items, total = VideoRepository(session).list_videos(
        limit=limit, offset=offset, status=status_filter, source=source, visible_to=viewer
    )
    return VideoList(items=[VideoSummary.model_validate(v) for v in items], total=total, limit=limit, offset=offset)


@router.get("/{video_id}", response_model=VideoDetail)
def get_video(video_id: str, session: SessionDep, storage: StorageDep, viewer: ViewerDep) -> VideoDetail:
    """Poll this to follow processing: status goes queued → (downloading) → processing (see `stage`) → ready | failed."""
    repo = VideoRepository(session)
    video = repo.get_visible(video_id, viewer)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    detail = VideoDetail.model_validate(video)
    detail.segment_counts = repo.segment_counts(video_id)
    if video.s3_key:
        detail.video_url = storage.presigned_url(video.s3_key)
    return detail


@router.get("/{video_id}/segments", response_model=SegmentList)
def list_segments(
    video_id: str,
    session: SessionDep,
    storage: StorageDep,
    viewer: ViewerDep,
    kind: str | None = Query(None, pattern="^(speech|visual)$"),
) -> SegmentList:
    repo = VideoRepository(session)
    if repo.get_visible(video_id, viewer) is None:
        raise HTTPException(status_code=404, detail="Video not found")
    items = []
    for segment in repo.list_segments(video_id, kind=kind):
        out = SegmentOut.model_validate(segment)
        if segment.frame_key:
            out.frame_url = storage.presigned_url(segment.frame_key)
        items.append(out)
    return SegmentList(video_id=video_id, items=items)


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: str, session: SessionDep, deletion: DeletionDep, viewer: ViewerDep) -> None:
    """Delete one of your own uploads: from search, storage and the database, right away. The shared library can't be
    deleted here (404, as for a video that doesn't exist, or one that isn't yours)."""
    video = VideoRepository(session).get(video_id)
    if video is None or video.owner_id is None or video.owner_id != viewer:
        raise HTTPException(status_code=404, detail="Video not found")
    session.close()  # the deletion uses its own session
    try:
        deletion.delete(video_id)
    except VideoBusy as exc:
        raise HTTPException(status_code=409, detail=f"Video is {exc}; delete it once processing finishes") from exc
