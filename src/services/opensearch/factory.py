from opensearchpy import OpenSearch

from src.config import Settings
from src.services.opensearch.client import OpenSearchService


def make_opensearch_service(settings: Settings) -> OpenSearchService:
    client = OpenSearch(
        hosts=[settings.opensearch_host],
        timeout=10,
        max_retries=2,
        retry_on_timeout=True,
        http_compress=True,
    )
    return OpenSearchService(client, alias=settings.opensearch_index)
