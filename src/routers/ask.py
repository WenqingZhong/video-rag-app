from celery.exceptions import TimeoutError as CeleryTimeout
from fastapi import APIRouter, HTTPException, status
from opensearchpy.exceptions import OpenSearchException

from src.dependencies import AskDep, StorageDep
from src.schemas.api.ask import AskRequest, AskResponse, ClipOut, IntentOut, Understood, UsageOut
from src.schemas.api.search import VideoRef
from src.services.metrics import observe_ask
from src.services.search.service import SearchUnavailable
from src.services.tracing import current_trace_id

router = APIRouter(tags=["Ask"])


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, service: AskDep, storage: StorageDep) -> AskResponse:
    """A plain request in, clip(s) out: understand (LLM + guards) → search → cut → template answer."""
    try:
        answer = service.ask(body.query, video_id=body.video_id, source=body.source, max_clips=body.max_clips)
    except (OpenSearchException, SearchUnavailable) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Search unavailable: {exc}") from exc
    except CeleryTimeout as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Clip cutting timed out") from exc

    clips = []
    for item in answer.clips:
        doc = item.hit.source
        clips.append(
            ClipOut(
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
        )
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
