from celery.result import AsyncResult
from fastapi import APIRouter, status

from src.schemas.api.search import JobStatus, ReindexRequest
from src.worker.celery_app import celery_app

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/reindex", response_model=JobStatus, status_code=status.HTTP_202_ACCEPTED)
def reindex(body: ReindexRequest | None = None) -> JobStatus:
    """Rebuild the search index from Postgres (all ready videos, or one). Runs on the worker."""
    from src.worker.tasks import rebuild_index

    job = rebuild_index.delay(body.video_id if body else None)
    return JobStatus(job_id=job.id, state=job.state)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    """State of any Celery job (PENDING / STARTED / SUCCESS / FAILURE) and its result."""
    job = AsyncResult(job_id, app=celery_app)
    result = job.result if job.successful() else (str(job.result) if job.failed() else None)
    return JobStatus(job_id=job_id, state=job.state, result=result)
