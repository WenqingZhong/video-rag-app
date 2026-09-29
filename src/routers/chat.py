import io
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.dependencies import ChatDep, SettingsDep, StorageDep
from src.schemas.api.chat import ChatResponse, ChatUpdateResponse
from src.services.metrics import observe_chat
from src.services.tracing import current_trace_id, discard_trace

router = APIRouter(tags=["Chat"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


@router.post("/chat", response_model=ChatResponse)
def chat(
    service: ChatDep,
    storage: StorageDep,
    settings: SettingsDep,
    message: Annotated[str, Form(max_length=500)] = "",
    conversation_id: Annotated[str | None, Form(max_length=64)] = None,
    image: Annotated[UploadFile | None, File(description="Optional photo: find the clip that looks like it")] = None,
    video: Annotated[UploadFile | None, File(description="Optional video: add it and ask for moments in it")] = None,
) -> ChatResponse:
    """One turn of a conversation. The agent picks a tool (find a clip, answer a question, search by image,
    fetch from Pexels after you agree, list videos) and replies from its result."""
    image_key = None
    if image is not None and image.filename:
        if not (image.content_type or "").startswith("image/"):
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Expected an image")
        data = image.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image larger than 10 MB")
        image_key = f"chat-images/{uuid.uuid4().hex}{Path(image.filename).suffix.lower() or '.jpg'}"
        storage.upload_fileobj(io.BytesIO(data), image_key, content_type=image.content_type)
    upload = None
    if video is not None and video.filename:
        if not (video.content_type or "").startswith("video/") and Path(video.filename).suffix.lower() not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Expected a video file")
        if video.size is not None and video.size > settings.upload_max_mb * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Video larger than {settings.upload_max_mb} MB"
            )
        upload = (video.file, video.filename, video.content_type, video.size)
    if not message.strip() and image_key is None and upload is None:
        raise HTTPException(status_code=422, detail="Send a message, an image or a video")

    turn = service.turn(conversation_id, message, image_key, upload)
    observe_chat(turn.action, turn.decided_by)
    return ChatResponse(request_id=current_trace_id(), **vars(turn))


@router.get("/chat/{conversation_id}/updates", response_model=ChatUpdateResponse)
def chat_updates(conversation_id: str, service: ChatDep) -> ChatUpdateResponse:
    """Poll after a Pexels download: "pending" while the videos are processed, then "done" with the clip.
    Cheap while nothing is happening (one Postgres lookup), so clients can call it every few seconds."""
    update = service.poll(conversation_id)
    if update.status in ("idle", "pending"):
        discard_trace()  # polls every few seconds: keep only the ones where something happened
    return ChatUpdateResponse(**vars(update))
