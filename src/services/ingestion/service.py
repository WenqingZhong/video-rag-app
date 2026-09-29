"""Getting videos *into* the system: store and queue. Heavy work is handed to the worker via Celery."""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.config import Settings
from src.models import Video, VideoSource, VideoStatus
from src.repositories import VideoRepository
from src.services.pexels import PexelsClient
from src.services.storage import StorageClient

logger = logging.getLogger(__name__)

# (video_id) -> Celery job id. Injected so the service doesn't import Celery and tests can fake it.
Enqueue = Callable[[str], str]


def raw_key(video_id: str, filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower() or ".mp4"
    return f"raw/{video_id}/source{suffix}"


@dataclass
class PexelsIngestResult:
    query: str
    requested: int
    queued: list[Video] = field(default_factory=list)
    skipped_existing: int = 0
    skipped_unsuitable: int = 0
    pages_searched: int = 0


class IngestionService:
    def __init__(
        self, session: Session, storage: StorageClient, settings: Settings, enqueue_process: Enqueue, enqueue_download: Enqueue
    ):
        self.session = session
        self.repo = VideoRepository(session)
        self.storage = storage
        self.settings = settings
        self.enqueue_process = enqueue_process
        self.enqueue_download = enqueue_download

    def _enqueue(self, video: Video, enqueue: Enqueue) -> None:
        # Enqueue only AFTER the row is committed: otherwise a fast worker could look for a row that isn't visible yet.
        try:
            video.job_id = enqueue(video.id)
        except Exception as exc:
            logger.exception("Failed to enqueue video %s", video.id)
            self.repo.set_status(video.id, VideoStatus.FAILED, error=f"enqueue failed: {exc}")
            self.session.commit()
            raise
        self.session.commit()

    def create_upload(
        self,
        fileobj: BinaryIO,
        filename: str | None,
        content_type: str | None,
        size_bytes: int | None,
        owner_id: str | None = None,
    ) -> Video:
        """`owner_id`: who uploaded it; only they can see it (None: shared library content)."""
        video_id = str(uuid.uuid4())
        key = raw_key(video_id, filename)
        # Bytes go to durable storage first; the queue only ever carries the id.
        self.storage.upload_fileobj(fileobj, key, content_type=content_type)
        video = self.repo.create(
            id=video_id,
            source=VideoSource.UPLOAD,
            title=Path(filename).stem if filename else None,
            original_filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            s3_key=key,
            owner_id=owner_id,
            status=VideoStatus.QUEUED,
        )
        self.session.commit()
        self._enqueue(video, self.enqueue_process)
        return video

    def reprocess(self, video: Video) -> Video:
        """Re-queue a video (e.g. after a failure). Safe because the pipeline replaces frames and segments."""
        enqueue = self.enqueue_process if video.s3_key else self.enqueue_download
        self.repo.set_status(video.id, VideoStatus.QUEUED)
        self.session.commit()
        self._enqueue(video, enqueue)
        return video

    def ingest_pexels(self, pexels: PexelsClient, query: str, count: int, max_pages: int = 5) -> PexelsIngestResult:
        """Queue up to `count` new Pexels videos for `query`, skipping ones we already have (idempotent)."""
        result = PexelsIngestResult(query=query, requested=count)
        page = 1
        while len(result.queued) < count and page <= max_pages:
            response = pexels.search_videos(query, per_page=min(max(count * 2, 15), 80), page=page)
            result.pages_searched = page
            known = self.repo.existing_source_ids(VideoSource.PEXELS, [str(v.id) for v in response.videos])

            for item in response.videos:
                if len(result.queued) >= count:
                    break
                if str(item.id) in known:
                    result.skipped_existing += 1
                    continue
                best = item.best_file(self.settings.pexels_max_height)
                if best is None or item.duration > self.settings.pexels_max_duration_sec:
                    result.skipped_unsuitable += 1
                    continue
                try:
                    with self.session.begin_nested():  # a concurrent run may insert the same video: skip, don't fail
                        video = self.repo.create(
                            source=VideoSource.PEXELS,
                            source_id=str(item.id),
                            title=item.title,
                            source_url=item.url,
                            source_file_url=best.link,
                            author_name=item.user.name,
                            author_url=item.user.url,
                            tags=item.tags,
                            content_type=best.file_type,
                            size_bytes=best.size,
                            duration_sec=float(item.duration),
                            width=best.width,
                            height=best.height,
                            fps=best.fps,
                            status=VideoStatus.QUEUED,
                        )
                except IntegrityError:
                    result.skipped_existing += 1
                    continue
                result.queued.append(video)

            if not response.next_page:
                break
            page += 1

        self.session.commit()
        for video in result.queued:
            self._enqueue(video, self.enqueue_download)
        logger.info(
            "Pexels '%s': queued %s, skipped %s existing / %s unsuitable (rate limit remaining: %s)",
            query, len(result.queued), result.skipped_existing, result.skipped_unsuitable, pexels.rate_limit_remaining,
        )  # fmt: skip
        return result
