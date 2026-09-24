from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.cache import CacheClient
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


SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
CacheDep = Annotated[CacheClient, Depends(get_cache_client)]
StorageDep = Annotated[StorageClient, Depends(get_storage_client)]
