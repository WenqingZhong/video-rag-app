import logging
from typing import Any

from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import NotFoundError, OpenSearchException

from src.services.opensearch.index_config import INDEX_BODY, versioned_index_name

logger = logging.getLogger(__name__)


class OpenSearchService:
    """Owns the segment index: creation (alias → versioned index), per-video (re)indexing, search, health."""

    def __init__(self, client: OpenSearch, alias: str):
        self.client = client
        self.alias = alias

    def ensure_index(self) -> str:
        """Create the versioned index and point the alias at it, if the alias doesn't exist yet. Idempotent."""
        if self.client.indices.exists_alias(name=self.alias):
            return self.alias
        index = versioned_index_name(self.alias)
        if not self.client.indices.exists(index=index):
            self.client.indices.create(index=index, body=INDEX_BODY)
            logger.info("Created index %s", index)
        self.client.indices.put_alias(index=index, name=self.alias)
        logger.info("Alias %s -> %s", self.alias, index)
        return self.alias

    def index_video(self, video_id: str, documents: list[dict[str, Any]], refresh: bool = True) -> int:
        """Replace all documents of one video: delete its old ones, then bulk-insert. Safe to repeat."""
        self.delete_video(video_id, refresh=False)
        if documents:
            actions = [{"_index": self.alias, "_id": doc["segment_id"], "_source": doc} for doc in documents]
            helpers.bulk(self.client, actions, refresh=False)
        if refresh:  # make the documents searchable immediately (default refresh is ~1 s)
            self.client.indices.refresh(index=self.alias)
        return len(documents)

    def delete_video(self, video_id: str, refresh: bool = True) -> int:
        try:
            response = self.client.delete_by_query(
                index=self.alias,
                body={"query": {"term": {"video_id": video_id}}},
                params={"refresh": "true" if refresh else "false", "conflicts": "proceed"},
            )
        except NotFoundError:
            return 0
        return response.get("deleted", 0)

    def search(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.client.search(index=self.alias, body=body)

    def count(self, video_id: str | None = None) -> int:
        body = {"query": {"term": {"video_id": video_id}}} if video_id else None
        return self.client.count(index=self.alias, body=body)["count"]

    def health_check(self) -> dict[str, Any]:
        try:
            cluster = self.client.cluster.health()
            if not self.client.indices.exists_alias(name=self.alias):
                return {"status": "unhealthy", "message": f"cluster {cluster['status']}, index alias '{self.alias}' missing"}
            return {"status": "healthy", "message": f"cluster {cluster['status']}, '{self.alias}' has {self.count()} segments"}
        except OpenSearchException as exc:
            return {"status": "unhealthy", "message": f"OpenSearch check failed: {exc}"}

    def close(self) -> None:
        self.client.close()
