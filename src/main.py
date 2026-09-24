import logging
import os
from contextlib import asynccontextmanager

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI

from src.config import get_settings
from src.db.factory import make_database
from src.routers import ping
from src.services.cache import make_cache_client
from src.services.storage import make_storage_client

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

    # Placeholders for later weeks
    app.state.opensearch_service = None
    app.state.llm_service = None

    logger.info("API ready")
    yield

    logger.info("Shutting down Video RAG API...")
    app.state.cache_client.close()
    app.state.database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="Video RAG API",
    description="Video-first RAG: retrieve clips from a video library by visual content or speech",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

app.include_router(ping.router, prefix=API_PREFIX)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": "Video RAG API is running", "docs": "/docs", "health": f"{API_PREFIX}/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="0.0.0.0")
