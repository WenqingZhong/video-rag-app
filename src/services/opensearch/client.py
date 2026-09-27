import logging
from collections.abc import Callable
from typing import Any

from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import NotFoundError, OpenSearchException

from src.services.opensearch.index_config import INDEX_VERSION, parse_version, versioned_index_name

logger = logging.getLogger(__name__)


class OpenSearchService:
    """Owns the segment index: creation (alias → versioned index), blue/green migration, indexing, search, health."""

    def __init__(self, client: OpenSearch, alias: str, index_body: dict, version: int = INDEX_VERSION):
        self.client = client
        self.alias = alias
        self.index_body = index_body
        self.version = version
        self._fields_cache: dict[str, set[str]] = {}

    # ---- lifecycle -------------------------------------------------------------------------------------
    def ensure_index(self) -> str:
        """Fresh install: create the current versioned index + alias. Existing alias: leave it (see migrate)."""
        if self.client.indices.exists_alias(name=self.alias):
            if self.is_outdated():
                logger.warning(
                    "Index %s is older than v%s: run POST /admin/reindex to migrate", self.alias_target(), self.version
                )
            return self.alias
        index = versioned_index_name(self.alias, self.version)
        if not self.client.indices.exists(index=index):
            self.client.indices.create(index=index, body=self.index_body)
            logger.info("Created index %s", index)
        self.client.indices.put_alias(index=index, name=self.alias)
        logger.info("Alias %s -> %s", self.alias, index)
        return self.alias

    def alias_target(self) -> str | None:
        try:
            return next(iter(self.client.indices.get_alias(name=self.alias)), None)
        except NotFoundError:
            return None

    def is_outdated(self) -> bool:
        target = self.alias_target()
        return target is not None and (parse_version(target) or 0) < self.version

    def migrate(self, fill: Callable[[str], None]) -> dict[str, Any]:
        """Blue/green: build the current version next to the live one, fill it, then switch the alias atomically.

        Searches keep hitting the old index until the switch; the old index is kept for rollback.
        """
        old = self.alias_target()
        new = versioned_index_name(self.alias, self.version)
        if old == new:
            raise ValueError(f"alias already points at {new}")
        if self.client.indices.exists(index=new):  # leftover from an interrupted migration: start clean
            self.client.indices.delete(index=new)
        self.client.indices.create(index=new, body=self.index_body)
        fill(new)
        self.client.indices.refresh(index=new)
        actions = [{"add": {"index": new, "alias": self.alias}}]
        if old:
            actions.insert(0, {"remove": {"index": old, "alias": self.alias}})
        self.client.indices.update_aliases(body={"actions": actions})
        logger.info("Alias %s switched %s -> %s", self.alias, old, new)
        return {"from": old, "to": new, "documents": self.count()}

    # ---- documents -------------------------------------------------------------------------------------
    def _fields(self, index: str) -> set[str]:
        """Top-level fields the target index knows (cached). Lets new code write safely into an older index."""
        if index not in self._fields_cache:
            mapping = self.client.indices.get_mapping(index=index)
            self._fields_cache[index] = set(next(iter(mapping.values()))["mappings"]["properties"])
        return self._fields_cache[index]

    def index_video(self, video_id: str, documents: list[dict[str, Any]], index: str | None = None, refresh: bool = True) -> int:
        """Replace all documents of one video: delete its old ones, then bulk-insert. Safe to repeat."""
        target = index or self.alias
        self.delete_video(video_id, index=target, refresh=False)
        if documents:
            known = self._fields(target)
            actions = [
                {
                    "_index": target,
                    "_id": doc["segment_id"],
                    "_source": {k: v for k, v in doc.items() if k in known and v is not None},
                }
                for doc in documents
            ]
            helpers.bulk(self.client, actions, refresh=False)
        if refresh:  # make the documents searchable immediately (default refresh is ~1 s)
            self.client.indices.refresh(index=target)
        return len(documents)

    def delete_video(self, video_id: str, index: str | None = None, refresh: bool = True) -> int:
        try:
            response = self.client.delete_by_query(
                index=index or self.alias,
                body={"query": {"term": {"video_id": video_id}}},
                params={"refresh": "true" if refresh else "false", "conflicts": "proceed"},
            )
        except NotFoundError:
            return 0
        return response.get("deleted", 0)

    # ---- queries ---------------------------------------------------------------------------------------
    def search(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.client.search(index=self.alias, body=body)

    def count(self, video_id: str | None = None) -> int:
        body = {"query": {"term": {"video_id": video_id}}} if video_id else None
        return self.client.count(index=self.alias, body=body)["count"]

    def health_check(self) -> dict[str, Any]:
        try:
            cluster = self.client.cluster.health()
            target = self.alias_target()
            if target is None:
                return {"status": "unhealthy", "message": f"cluster {cluster['status']}, index alias '{self.alias}' missing"}
            if self.is_outdated():
                return {
                    "status": "unhealthy",
                    "message": f"'{self.alias}' → {target} is older than v{self.version}: POST /admin/reindex",
                }
            return {
                "status": "healthy",
                "message": f"cluster {cluster['status']}, '{self.alias}' → {target}, {self.count()} segments",
            }
        except OpenSearchException as exc:
            return {"status": "unhealthy", "message": f"OpenSearch check failed: {exc}"}

    def close(self) -> None:
        self.client.close()
