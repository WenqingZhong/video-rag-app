from typing import Annotated

from celery.exceptions import TimeoutError as CeleryTimeout
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from opensearchpy.exceptions import OpenSearchException

from src.dependencies import AskDep, ImageAskDep, QADep, StorageDep
from src.schemas.api.ask import (
    AnswerRequest,
    AnswerResponse,
    AskRequest,
    AskResponse,
    CitationOut,
    ClipOut,
    ImageAskResponse,
    IntentOut,
    Understood,
    UsageOut,
)
from src.schemas.api.search import VideoRef
from src.services.metrics import observe_ask
from src.services.search.service import SearchUnavailable
from src.services.tracing import current_trace_id

router = APIRouter(tags=["Ask"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, service: AskDep, storage: StorageDep) -> AskResponse:
    """A plain request in, clip(s) out: understand (LLM + guards) → search → cut → template answer."""
    try:
        answer = service.ask(body.query, video_id=body.video_id, source=body.source, max_clips=body.max_clips)
    except (OpenSearchException, SearchUnavailable) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Search unavailable: {exc}") from exc
    except CeleryTimeout as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Clip cutting timed out") from exc

    clips = [_clip_out(item, storage) for item in answer.clips]
    understood = answer.understanding
    observe_ask(answer.status, understood.source, answer.cache, answer.timings)
    return AskResponse(
        request_id=current_trace_id(),
        query=body.query,
        status=answer.status,
        answer=answer.answer,
        understood=Understood(
            intent=IntentOut(**understood.intent.model_dump()), source=understood.source, notes=understood.notes
        ),
        strategy=answer.strategy,
        clips=clips,
        timings=answer.timings,
        usage=_usage(understood),
        cache=answer.cache,
    )


def _usage(understood) -> UsageOut:
    call = understood.llm_call
    if call is None:
        avoided = understood.avoided_call  # a cache hit: nothing spent, this much saved
        if avoided is None:
            return UsageOut()
        return UsageOut(saved_tokens=avoided.total_tokens, saved_cost_usd=understood.saved_cost_usd)
    return UsageOut(
        prompt_tokens=call.prompt_tokens,
        output_tokens=call.output_tokens,
        total_tokens=call.total_tokens,
        cost_usd=understood.cost_usd,
        outcome=understood.llm_outcome,
    )


def _clip_out(item, storage) -> ClipOut:
    doc = item.hit.source
    return ClipOut(
        url=storage.presigned_url(item.key),
        start_sec=item.clip.start_sec,
        end_sec=item.clip.end_sec,
        duration_sec=item.clip.duration,
        cached=item.cached,
        video=VideoRef(
            id=doc["video_id"],
            title=doc.get("video_title"),
            source=doc.get("video_source"),
            author=doc.get("video_author"),
            source_url=doc.get("video_source_url"),
            duration_sec=doc.get("video_duration_sec"),
        ),
        kind=doc["kind"],
        matched_text=item.hit.matched_text,
        match_score=item.hit.match_score,
        caption=doc.get("caption"),
        explanation=item.explanation,
    )


@router.post("/ask/image", response_model=ImageAskResponse)
def ask_image(
    service: ImageAskDep,
    storage: StorageDep,
    file: Annotated[UploadFile, File(description="A photo or screenshot (JPEG/PNG)")],
    video_id: Annotated[str | None, Form()] = None,
    max_clips: Annotated[int, Form(ge=1, le=3)] = 1,
) -> ImageAskResponse:
    """A photo in, the clip that looks most like it out (CLIP image vs keyframe vectors)."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Expected an image")
    image = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(image) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image larger than 10 MB")
    try:
        answer = service.ask(image, video_id=video_id, max_clips=max_clips)
    except (OpenSearchException, SearchUnavailable) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Search unavailable: {exc}") from exc
    except CeleryTimeout as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Clip cutting timed out") from exc
    return ImageAskResponse(
        request_id=current_trace_id(),
        status=answer.status,
        answer=answer.answer,
        clips=[_clip_out(c, storage) for c in answer.clips],
        timings=answer.timings,
    )


@router.post("/answer", response_model=AnswerResponse)
def answer_question(body: AnswerRequest, service: QADep, storage: StorageDep) -> AnswerResponse:
    """A question about what videos say or show, answered only from their transcripts and captions, with citations."""
    try:
        result = service.answer(body.question, video_id=body.video_id)
    except (OpenSearchException, SearchUnavailable) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Search unavailable: {exc}") from exc
    except CeleryTimeout as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Clip cutting timed out") from exc
    call = result.llm_call
    usage = UsageOut()
    if call is not None:
        usage = UsageOut(prompt_tokens=call.prompt_tokens, output_tokens=call.output_tokens, total_tokens=call.total_tokens,
                         cost_usd=result.cost_usd, outcome=result.llm_outcome)  # fmt: skip
    return AnswerResponse(
        request_id=current_trace_id(),
        question=body.question,
        status=result.status,
        answer=result.answer,
        citations=[CitationOut(**vars(e)) for e in result.citations],
        clip_url=storage.presigned_url(result.clip_key) if result.clip_key else None,
        notes=result.notes,
        usage=usage,
        timings=result.timings,
    )
