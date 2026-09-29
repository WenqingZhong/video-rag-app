"""Deleting a video everywhere it lives: the search index, S3 (source, keyframes, clips) and Postgres.

Order matters: the index first, so the video stops appearing in results even if a later step fails; Postgres last,
so a failed run can simply be retried (the row still says what to clean up). Then the index version is bumped, so no
cached answer can still point at the deleted video.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from src.db.interfaces.base import BaseDatabase
from src.models import VideoStatus
from src.repositories import VideoRepository
from src.services.cache import IndexVersion
from src.services.opensearch import OpenSearchService
from src.services.storage import StorageClient
from src.services.tracing import span

logger = logging.getLogger(__name__)

BUSY = (VideoStatus.DOWNLOADING, VideoStatus.PROCESSING)  # a worker is writing its files: delete after it finishes


def prefixes(video_id: str) -> list[str]:
    return [f"raw/{video_id}/", f"frames/{video_id}/", f"clips/{video_id}/"]


@dataclass
class Deleted:
    video_id: str
    title: str | None
    documents: int
    files: int


class VideoBusy(Exception):
    """The video is being downloaded or processed."""


class DeletionService:
    def __init__(
        self,
        database: BaseDatabase,
        storage: StorageClient,
        opensearch: OpenSearchService,
        versions: IndexVersion | None,
    ):
        self.database = database
        self.storage = storage
        self.opensearch = opensearch
        self.versions = versions

    def delete(self, video_id: str) -> Deleted | None:
        """None if there's no such video. Raises VideoBusy while a worker is on it. Ownership is the caller's check."""
        with span("video.delete", video_id=video_id) as step, self.database.get_session() as session:
            repo = VideoRepository(session)
            video = repo.get(video_id)
            if video is None:
                return None
            if video.status in BUSY:
                raise VideoBusy(video.status)
            title = video.title or video.original_filename
            documents = self.opensearch.delete_video(video_id)
            files = sum(self.storage.delete_prefix(prefix) for prefix in prefixes(video_id))
            repo.delete_video(video_id)
            session.commit()
            step.set(documents=documents, files=files)
        if self.versions is not None:
            self.versions.bump()
        logger.info("deleted video %s (%d documents, %d files)", video_id, documents, files)
        return Deleted(video_id, title, documents, files)

    def delete_expired(self, days: int) -> list[Deleted]:
        """Users' uploads older than `days` (the shared library never expires). Busy ones wait for the next run."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        with self.database.get_session() as session:
            ids = [v.id for v in VideoRepository(session).expired_uploads(cutoff)]
        deleted = []
        for video_id in ids:
            try:
                if (result := self.delete(video_id)) is not None:
                    deleted.append(result)
            except VideoBusy:
                continue
        return deleted
