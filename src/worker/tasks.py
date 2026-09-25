import logging
import tempfile
from pathlib import Path

import httpx
from botocore.exceptions import BotoCoreError, ClientError
from celery import Task

from src.config import get_settings
from src.models import VideoStatus
from src.repositories import VideoRepository
from src.services.ingestion.service import raw_key
from src.services.processing.pipeline import VideoPipeline
from src.worker.celery_app import celery_app
from src.worker.context import get_database, get_storage, get_transcriber

logger = logging.getLogger(__name__)

# Network hiccups are worth retrying; bad input (corrupt file, missing row) is not.
TRANSIENT_ERRORS = (httpx.TransportError, BotoCoreError, ClientError, ConnectionError, TimeoutError)
MAX_RETRIES = 3


def _mark_failed(video_id: str, error: str) -> None:
    with get_database().get_session() as session:
        VideoRepository(session).set_status(video_id, VideoStatus.FAILED, error=error[:2000])
        session.commit()


def _retry_or_fail(task: Task, video_id: str, exc: Exception):
    if isinstance(exc, TRANSIENT_ERRORS) and task.request.retries < MAX_RETRIES:
        countdown = 10 * 2**task.request.retries
        logger.warning(
            "%s(%s) transient error, retry %s in %ss: %s", task.name, video_id, task.request.retries + 1, countdown, exc
        )
        raise task.retry(exc=exc, countdown=countdown)
    logger.exception("%s(%s) failed", task.name, video_id)
    _mark_failed(video_id, f"{type(exc).__name__}: {exc}")
    raise exc


@celery_app.task(name="system.ping")
def ping() -> str:
    """Round-trip task used to verify the broker -> worker -> result backend path."""
    return "pong"


@celery_app.task(name="video.download_pexels", bind=True, max_retries=MAX_RETRIES)
def download_pexels(self: Task, video_id: str) -> dict:
    """Stream the chosen Pexels rendition into S3, then hand over to video.process."""
    try:
        with get_database().get_session() as session:
            repo = VideoRepository(session)
            video = repo.get(video_id)
            if video is None:
                raise ValueError(f"video {video_id} not found")
            url = video.source_file_url
            repo.set_status(video_id, VideoStatus.DOWNLOADING, stage="downloading")
            session.commit()

        key = raw_key(video_id, "source.mp4")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.mp4"
            with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30, read=120)) as response:
                response.raise_for_status()
                with path.open("wb") as fh:
                    for chunk in response.iter_bytes(1024 * 1024):
                        fh.write(chunk)
            size = path.stat().st_size
            get_storage().upload_file(path, key, content_type="video/mp4")

        with get_database().get_session() as session:
            repo = VideoRepository(session)
            repo.update(video_id, s3_key=key, size_bytes=size, content_type="video/mp4", status=VideoStatus.QUEUED, stage=None)
            session.commit()  # commit BEFORE enqueueing, so the processing task always sees s3_key
            job = process_video.delay(video_id)
            repo.update(video_id, job_id=job.id)
            session.commit()
        return {"video_id": video_id, "s3_key": key, "size_bytes": size, "process_job_id": job.id}
    except Exception as exc:  # noqa: BLE001 - classified in _retry_or_fail
        _retry_or_fail(self, video_id, exc)


@celery_app.task(name="video.process", bind=True, max_retries=MAX_RETRIES)
def process_video(self: Task, video_id: str) -> dict:
    """The Phase 2 pipeline: scenes, keyframes, transcript -> segments in Postgres, frames in S3."""
    try:
        pipeline = VideoPipeline(get_database(), get_storage(), get_settings(), transcriber_factory=get_transcriber)
        return pipeline.process(video_id)
    except Exception as exc:  # noqa: BLE001 - classified in _retry_or_fail
        _retry_or_fail(self, video_id, exc)
