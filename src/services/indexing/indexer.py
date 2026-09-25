"""Copy one video's segments from Postgres (source of truth) into OpenSearch (derived, rebuildable)."""

from src.db.interfaces.base import BaseDatabase
from src.repositories import VideoRepository
from src.services.indexing.documents import segment_documents
from src.services.opensearch import OpenSearchService


def index_video(database: BaseDatabase, opensearch: OpenSearchService, video_id: str) -> int:
    with database.get_session() as session:
        repo = VideoRepository(session)
        video = repo.get(video_id)
        if video is None:
            raise ValueError(f"video {video_id} not found")
        documents = segment_documents(video, repo.list_segments(video_id))
    return opensearch.index_video(video_id, documents)
