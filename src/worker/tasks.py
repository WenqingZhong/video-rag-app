import logging
import tempfile
from pathlib import Path

import httpx
from botocore.exceptions import BotoCoreError, ClientError
from celery import Task
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import ConnectionTimeout as OpenSearchTimeout

from src.config import get_settings
from src.models import VideoStatus
from src.repositories import VideoRepository
from src.services.clips import ClipRange, clip_key
from src.services.indexing import index_video
from src.services.ingestion.service import raw_key
from src.services.processing import ffmpeg
from src.services.processing.pipeline import VideoPipeline
from src.worker.celery_app import celery_app
from src.worker.context import get_database, get_enricher, get_opensearch, get_storage, get_transcriber

logger = logging.getLogger(__name__)

# Network hiccups are worth retrying; bad input (corrupt file, missing row) is not.
TRANSIENT_ERRORS = (
    httpx.TransportError,
    BotoCoreError,
    ClientError,
    ConnectionError,
    TimeoutError,
    OpenSearchConnectionError,
    OpenSearchTimeout,
)
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
        pipeline = VideoPipeline(
            get_database(),
            get_storage(),
            get_settings(),
            transcriber_factory=get_transcriber,
            indexer=lambda vid: index_video(get_database(), get_opensearch(), vid),
            enricher=get_enricher(),
        )
        return pipeline.process(video_id)
    except Exception as exc:  # noqa: BLE001 - classified in _retry_or_fail
        _retry_or_fail(self, video_id, exc)


@celery_app.task(name="video.enrich_visual", bind=True, max_retries=MAX_RETRIES)
def enrich_visual(self: Task, video_id: str, force: bool = False) -> dict:
    """Add CLIP vectors + captions to an already-processed video, from the keyframes stored in S3.

    Much cheaper than reprocessing (no download, scene detection or Whisper). The video stays `ready`
    and searchable throughout; only its visual segments are updated, then the video is re-indexed.
    """
    try:
        with get_database().get_session() as session:
            segments = VideoRepository(session).list_segments(video_id, kind="visual")
            todo = [
                (s.id, s.frame_key) for s in segments if s.frame_key and (force or s.image_embedding is None or s.caption is None)
            ]
        if not todo:
            return {"video_id": video_id, "enriched": 0}

        images = [get_storage().get_bytes(key) for _, key in todo]
        fields = get_enricher().enrich(images)
        with get_database().get_session() as session:
            repo = VideoRepository(session)
            for (segment_id, _), values in zip(todo, fields, strict=True):
                repo.update_segment(segment_id, **values)
            session.commit()
        indexed = index_video(get_database(), get_opensearch(), video_id)
        return {"video_id": video_id, "enriched": len(todo), "indexed": indexed}
    except Exception as exc:
        # Unlike processing, a failed enrichment must NOT mark a ready video as failed: it stays searchable as before.
        if isinstance(exc, TRANSIENT_ERRORS) and self.request.retries < MAX_RETRIES:
            raise self.retry(exc=exc, countdown=10 * 2**self.request.retries) from exc
        logger.exception("enrich_visual(%s) failed", video_id)
        raise


@celery_app.task(name="index.rebuild", bind=True)
def rebuild_index(self: Task, video_id: str | None = None) -> dict:
    """Re-copy segments from Postgres into OpenSearch: every ready video, or just one.

    If the live index is an older version than the code's mapping, this is a blue/green migration:
    build the new versioned index, fill it, then switch the alias atomically (old index kept for rollback).
    """
    with get_database().get_session() as session:
        repo = VideoRepository(session)
        if video_id:
            ids = [video_id]
        else:
            videos, _ = repo.list_videos(limit=100_000, status=VideoStatus.READY)
            ids = [v.id for v in videos]

    opensearch = get_opensearch()
    indexed, failed = 0, []

    def fill(index: str | None = None) -> None:
        nonlocal indexed
        for vid in ids:
            try:
                indexed += index_video(get_database(), opensearch, vid, index=index)
            except Exception as exc:  # one bad video must not abort the rebuild
                logger.exception("rebuild: indexing %s failed", vid)
                failed.append({"video_id": vid, "error": f"{type(exc).__name__}: {exc}"})

    migration = None
    if video_id is None and opensearch.is_outdated():
        migration = opensearch.migrate(fill)
    else:
        fill()
    return {
        "videos": len(ids) - len(failed),
        "segments": indexed,
        "failed": failed,
        "index_total": opensearch.count(),
        "migration": migration,
    }


@celery_app.task(name="clip.cut", bind=True, max_retries=2)
def cut_clip(self: Task, video_id: str, start_sec: float, end_sec: float) -> str:
    """Cut one clip into S3 (clips/…) and return its key. Routed to the `clips` queue (see celery_app)."""
    clip = ClipRange(start_sec, end_sec)
    key = clip_key(video_id, clip)
    storage = get_storage()
    if storage.exists(key):  # another request cut it in the meantime
        return key
    try:
        with get_database().get_session() as session:
            video = VideoRepository(session).get(video_id)
            if video is None or not video.s3_key:
                raise ValueError(f"video {video_id} has no stored file")
            source_key = video.s3_key
        with tempfile.TemporaryDirectory(prefix="clip-") as tmp:
            source = storage.download_file(source_key, Path(tmp) / Path(source_key).name)
            out = ffmpeg.cut_clip(source, clip.start_sec, clip.end_sec, Path(tmp) / "clip.mp4")
            storage.upload_file(out, key, content_type="video/mp4")
        return key
    except TRANSIENT_ERRORS as exc:
        raise self.retry(exc=exc, countdown=2) from exc
