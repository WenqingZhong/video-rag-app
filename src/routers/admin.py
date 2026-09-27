from datetime import UTC, datetime, timedelta

from celery.result import AsyncResult
from fastapi import APIRouter, status

from src.dependencies import SessionDep, SettingsDep
from src.models import VideoStatus
from src.repositories import VideoRepository
from src.schemas.api.search import EnrichRequest, EnrichResponse, JobStatus, ReindexRequest
from src.schemas.api.traces import CleanupRequest, CleanupResponse
from src.services.tracing import delete_spans_before
from src.worker.celery_app import celery_app

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/reindex", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def reindex(body: ReindexRequest | None = None) -> JobStatus:
    """Rebuild the search index from Postgres (all ready videos, or one). Runs on the worker."""
    from src.worker.tasks import rebuild_index

    job = rebuild_index.delay(body.video_id if body else None)
    return JobStatus(job_id=job.id, state=job.state)


@router.post("/enrich", response_model=EnrichResponse, status_code=status.HTTP_202_ACCEPTED)
def enrich(session: SessionDep, body: EnrichRequest | None = None) -> EnrichResponse:
    """Backfill CLIP vectors + captions for already-processed videos (one job per video; keyframes come from S3)."""
    from src.worker.tasks import enrich_visual

    body = body or EnrichRequest()
    if body.video_id:
        ids = [body.video_id]
    else:
        videos, _ = VideoRepository(session).list_videos(limit=100_000, status=VideoStatus.READY)
        ids = [v.id for v in videos]
    jobs = [enrich_visual.delay(video_id, force=body.force).id for video_id in ids]
    return EnrichResponse(queued=len(jobs), job_ids=jobs)


@router.post("/cleanup-traces", response_model=CleanupResponse)
def cleanup_traces(session: SessionDep, settings: SettingsDep, body: CleanupRequest | None = None) -> CleanupResponse:
    """Delete trace spans older than the retention period (called daily by the Airflow "maintenance" DAG)."""
    days = (body.older_than_days if body else None) or settings.trace_retention_days
    before = datetime.now(UTC) - timedelta(days=days)
    return CleanupResponse(deleted_spans=delete_spans_before(session, before), before=before)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    """State of any Celery job (PENDING / STARTED / SUCCESS / FAILURE) and its result."""
    job = AsyncResult(job_id, app=celery_app)
    result = job.result if job.successful() else (str(job.result) if job.failed() else None)
    return JobStatus(job_id=job_id, state=job.state, result=result)
