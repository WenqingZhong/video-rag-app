from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.cache import CacheClient
from src.services.ingestion import IngestionService
from src.services.pexels import PexelsClient
from src.services.storage import StorageClient


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    with database.get_session() as session:
        yield session


def get_cache_client(request: Request) -> CacheClient:
    return request.app.state.cache_client


def get_storage_client(request: Request) -> StorageClient:
    return request.app.state.storage_client


def get_pexels_client(request: Request) -> PexelsClient:
    client = getattr(request.app.state, "pexels_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Pexels is not configured (set PEXELS_API_KEY)")
    return client


def _enqueue_process(video_id: str) -> str:
    from src.worker.tasks import process_video

    return process_video.delay(video_id).id


def _enqueue_download(video_id: str) -> str:
    from src.worker.tasks import download_pexels

    return download_pexels.delay(video_id).id


def get_ingestion_service(
    session: Annotated[Session, Depends(get_db_session)],
    storage: Annotated[StorageClient, Depends(get_storage_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestionService:
    return IngestionService(session, storage, settings, enqueue_process=_enqueue_process, enqueue_download=_enqueue_download)


SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
CacheDep = Annotated[CacheClient, Depends(get_cache_client)]
StorageDep = Annotated[StorageClient, Depends(get_storage_client)]
PexelsDep = Annotated[PexelsClient, Depends(get_pexels_client)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion_service)]
