import logging
import os
from contextlib import asynccontextmanager

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI
from opensearchpy.exceptions import OpenSearchException

from src.config import get_settings
from src.db.factory import make_database
from src.routers import admin, ask, clips, ping, search, videos
from src.services.cache import make_cache_client
from src.services.captioning import make_captioner
from src.services.embeddings import make_embedding_client
from src.services.opensearch import make_opensearch_service
from src.services.pexels import make_pexels_client
from src.services.storage import make_storage_client
from src.services.understanding import make_query_understanding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared clients once per process. Every dependency degrades gracefully if down."""
    logger.info("Starting Video RAG API...")

    settings = get_settings()
    app.state.settings = settings
    app.state.database = make_database()
    app.state.cache_client = make_cache_client(settings)
    app.state.storage_client = make_storage_client(settings)

    try:
        app.state.storage_client.ensure_bucket()
        logger.info("Object storage bucket '%s' ready", settings.s3_bucket)
    except (BotoCoreError, ClientError) as exc:
        logger.warning("Object storage unavailable during startup: %s", exc)

    app.state.pexels_client = make_pexels_client(settings) if settings.pexels_api_key else None
    if app.state.pexels_client is None:
        logger.warning("PEXELS_API_KEY not set: Pexels ingestion endpoints will return 503")

    app.state.opensearch_service = make_opensearch_service(settings)
    try:
        app.state.opensearch_service.ensure_index()
        logger.info("Search index alias '%s' ready", settings.opensearch_index)
    except OpenSearchException as exc:
        logger.warning("OpenSearch unavailable during startup: %s", exc)

    # Clients only: no network call at startup, so the API boots even if these services are still loading.
    app.state.embedding_client = make_embedding_client(settings)
    app.state.captioner = make_captioner(settings)
    app.state.understanding = make_query_understanding(settings)  # LLM client only: no call at startup

    # Placeholder for later weeks
    app.state.llm_service = None

    logger.info("API ready")
    yield

    logger.info("Shutting down Video RAG API...")
    app.state.cache_client.close()
    app.state.opensearch_service.close()
    app.state.embedding_client.close()
    app.state.captioner.close()
    if app.state.understanding.llm is not None:
        app.state.understanding.llm.close()
    if app.state.pexels_client is not None:
        app.state.pexels_client.close()
    app.state.database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="Video RAG API",
    description="Video-first RAG: retrieve clips from a video library by visual content or speech",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

app.include_router(ping.router, prefix=API_PREFIX)
app.include_router(videos.router, prefix=API_PREFIX)
app.include_router(search.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)
app.include_router(clips.router, prefix=API_PREFIX)
app.include_router(ask.router, prefix=API_PREFIX)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": "Video RAG API is running", "docs": "/docs", "health": f"{API_PREFIX}/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="0.0.0.0")
