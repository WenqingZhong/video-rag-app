from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.repositories import VideoRepository
from src.services.agent import ChatService, Toolbox
from src.services.answering import AskService, ImageAskService
from src.services.cache import CacheClient
from src.services.captioning import Captioner
from src.services.clips import ClipService
from src.services.embeddings import EmbeddingClient
from src.services.identity import ensure_viewer, is_admin
from src.services.ingestion import DeletionService, IngestionService
from src.services.limits import Limiter, LimitExceeded, Principal, current_principal
from src.services.opensearch import OpenSearchService
from src.services.pexels import PexelsClient
from src.services.qa import QAService
from src.services.search import SearchService
from src.services.storage import StorageClient
from src.services.understanding import QueryUnderstanding


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    with database.get_session() as session:
        yield session


def get_viewer(request: Request, response: Response, settings: Annotated[Settings, Depends(get_settings)]) -> str:
    """Who is asking (see services/identity.py). A first-time web visitor gets an anonymous session cookie."""
    viewer = ensure_viewer(request, response, settings)
    principal = current_principal()
    if principal is not None:
        principal.viewer = viewer  # from now on, model calls in this request count against this viewer
    return viewer


def require_admin(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> None:
    """Operator-only endpoints (admin jobs, traces, bulk ingestion): the x-admin-token header."""
    if not is_admin(request, settings):
        raise HTTPException(status_code=403, detail="Admin token required")


def get_limiter(request: Request) -> Limiter | None:
    return getattr(request.app.state, "limiter", None)


def principal_for(viewer: str) -> Principal:
    return current_principal() or Principal(viewer=viewer)


def too_many(exc: LimitExceeded) -> HTTPException:
    """429 with the reason in words: clients show `detail` to the user as it is."""
    return HTTPException(status_code=429, detail=exc.message, headers={"Retry-After": str(exc.retry_after)})


def check_limits(viewer: Annotated[str, Depends(get_viewer)], limiter: Annotated[Limiter | None, Depends(get_limiter)]) -> None:
    """Before a request that may call the model: the request rate, and the day's tokens (limits.py)."""
    if limiter is not None:
        try:
            limiter.check_request(principal_for(viewer))
        except LimitExceeded as exc:
            raise too_many(exc) from exc


def get_search_scope(
    viewer: Annotated[str, Depends(get_viewer)], database: Annotated[BaseDatabase, Depends(get_database)]
) -> str | None:
    """The viewer if they own videos, else None (the shared library). Viewers without uploads see the same results,
    so they share one answer cache instead of each filling their own."""
    with database.get_session() as session:
        return viewer if VideoRepository(session).owns_any(viewer) else None


def get_cache_client(request: Request) -> CacheClient:
    return request.app.state.cache_client


def get_storage_client(request: Request) -> StorageClient:
    return request.app.state.storage_client


def get_pexels_client(request: Request) -> PexelsClient:
    client = getattr(request.app.state, "pexels_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Pexels is not configured (set PEXELS_API_KEY)")
    return client


def get_opensearch_service(request: Request) -> OpenSearchService:
    service = getattr(request.app.state, "opensearch_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Search is not available (OpenSearch not initialised)")
    return service


def get_embedding_client(request: Request) -> EmbeddingClient | None:
    return getattr(request.app.state, "embedding_client", None)


def get_captioner(request: Request) -> Captioner | None:
    return getattr(request.app.state, "captioner", None)


def get_search_service(
    opensearch: Annotated[OpenSearchService, Depends(get_opensearch_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[EmbeddingClient | None, Depends(get_embedding_client)],
    scope: Annotated[str | None, Depends(get_search_scope)],
) -> SearchService:
    return SearchService(opensearch, settings, embedder=embedder, viewer=scope)


def _cut_on_clip_worker(timeout: float):
    def cut(video_id: str, start_sec: float, end_sec: float) -> str:
        from src.worker.tasks import cut_clip

        # Someone is waiting: block (in FastAPI's thread pool) until the clip worker returns the S3 key.
        return cut_clip.delay(video_id, start_sec, end_sec).get(timeout=timeout)

    return cut


def get_clip_service(
    storage: Annotated[StorageClient, Depends(get_storage_client)], settings: Annotated[Settings, Depends(get_settings)]
) -> ClipService:
    return ClipService(storage, cut=_cut_on_clip_worker(settings.clip_timeout))


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


def get_deletion_service(
    database: Annotated[BaseDatabase, Depends(get_database)],
    storage: Annotated[StorageClient, Depends(get_storage_client)],
    opensearch: Annotated[OpenSearchService, Depends(get_opensearch_service)],
    request: Request,
) -> DeletionService:
    return DeletionService(database, storage, opensearch, getattr(request.app.state, "index_version", None))


SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
CacheDep = Annotated[CacheClient, Depends(get_cache_client)]
StorageDep = Annotated[StorageClient, Depends(get_storage_client)]
PexelsDep = Annotated[PexelsClient, Depends(get_pexels_client)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion_service)]
OpenSearchDep = Annotated[OpenSearchService, Depends(get_opensearch_service)]
SearchDep = Annotated[SearchService, Depends(get_search_service)]
EmbedderDep = Annotated[EmbeddingClient | None, Depends(get_embedding_client)]
CaptionerDep = Annotated[Captioner | None, Depends(get_captioner)]
ClipDep = Annotated[ClipService, Depends(get_clip_service)]
ViewerDep = Annotated[str, Depends(get_viewer)]
DeletionDep = Annotated[DeletionService, Depends(get_deletion_service)]
LimiterDep = Annotated[Limiter | None, Depends(get_limiter)]
Limited = Depends(check_limits)  # for routes: dependencies=[Limited]
AdminOnly = Depends(require_admin)


def get_query_understanding(request: Request) -> QueryUnderstanding:
    return request.app.state.understanding


def get_ask_service(
    understanding: Annotated[QueryUnderstanding, Depends(get_query_understanding)],
    search: Annotated[SearchService, Depends(get_search_service)],
    clips: Annotated[ClipService, Depends(get_clip_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    request: Request,
) -> AskService:
    answers = getattr(request.app.state, "answer_cache", None)
    return AskService(understanding, search, clips, settings, answers=answers, scope=search.viewer)


def get_image_ask_service(
    search: Annotated[SearchService, Depends(get_search_service)],
    clips: Annotated[ClipService, Depends(get_clip_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ImageAskService:
    return ImageAskService(search, clips, settings)


def get_qa_service(
    request: Request,
    database: Annotated[BaseDatabase, Depends(get_database)],
    search: Annotated[SearchService, Depends(get_search_service)],
    clips: Annotated[ClipService, Depends(get_clip_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> QAService:
    state = request.app.state
    return QAService(
        database, search, getattr(state, "answerer", None), settings, getattr(state, "usage_recorder", None), clips=clips
    )


def get_chat_service(
    request: Request,
    database: Annotated[BaseDatabase, Depends(get_database)],
    storage: Annotated[StorageClient, Depends(get_storage_client)],
    ask: Annotated[AskService, Depends(get_ask_service)],
    image_ask: Annotated[ImageAskService, Depends(get_image_ask_service)],
    qa: Annotated[QAService, Depends(get_qa_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    viewer: Annotated[str, Depends(get_viewer)],
    deletion: Annotated[DeletionService, Depends(get_deletion_service)],
) -> ChatService:
    state = request.app.state

    def make_ingestion(session: Session) -> IngestionService:
        return IngestionService(session, storage, settings, enqueue_process=_enqueue_process, enqueue_download=_enqueue_download)

    toolbox = Toolbox(
        database,
        storage,
        ask,
        image_ask,
        qa,
        make_ingestion,
        getattr(state, "pexels_client", None),
        viewer,
        deletion,
        getattr(state, "limiter", None),
    )
    return ChatService(
        toolbox,
        state.conversations,
        getattr(state, "router", None),
        getattr(state, "usage_recorder", None),
        settings.chat_fetch_timeout_sec,
        viewer=viewer,
    )


UnderstandingDep = Annotated[QueryUnderstanding, Depends(get_query_understanding)]
ImageAskDep = Annotated[ImageAskService, Depends(get_image_ask_service)]
QADep = Annotated[QAService, Depends(get_qa_service)]
ChatDep = Annotated[ChatService, Depends(get_chat_service)]
AskDep = Annotated[AskService, Depends(get_ask_service)]
