from fastapi import APIRouter, HTTPException, status
from opensearchpy.exceptions import OpenSearchException

from src.dependencies import SearchDep, StorageDep, UnderstandingDep
from src.schemas.api.search import SearchHitOut, SearchRequest, SearchResponse, VideoRef
from src.services.search import SearchResult
from src.services.search.query_parser import ParsedQuery
from src.services.search.service import SearchUnavailable

router = APIRouter(tags=["Search"])

PLAY_PADDING_SEC = 1.0  # a little context before/after the matched words


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, service: SearchDep, storage: StorageDep, understanding: UnderstandingDep) -> SearchResponse:
    """Quoted text → phrase search over transcripts (exact → slop → fuzzy).
    Otherwise hybrid: keyword (BM25 over transcripts, captions, titles) + vector (CLIP over keyframes), fused with RRF."""
    try:
        if body.understand:
            intent = understanding.understand(body.query).intent
            if not intent.has_subject:  # nothing to search for (as /ask would ask back)
                result = SearchResult(parsed=ParsedQuery(body.query, None, body.query), strategy="none")
            else:
                result = service.search_intent(
                    intent, video_id=body.video_id, source=body.source, size=body.size, group_by_video=body.group_by_video
                )
        else:
            result = service.search(
                body.query,
                video_id=body.video_id,
                kind=body.kind,
                source=body.source,
                size=body.size,
                group_by_video=body.group_by_video,
                mode=body.mode,
            )
    except (OpenSearchException, SearchUnavailable) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Search unavailable: {exc}") from exc

    hits = []
    for hit in result.hits:
        doc = hit.source
        play_url = None
        if doc.get("video_s3_key"):
            start = max(hit.match_start_sec - PLAY_PADDING_SEC, 0)
            end = hit.match_end_sec + PLAY_PADDING_SEC
            play_url = f"{storage.presigned_url(doc['video_s3_key'])}#t={start:.2f},{end:.2f}"
        hits.append(
            SearchHitOut(
                video=VideoRef(
                    id=doc["video_id"],
                    title=doc.get("video_title"),
                    source=doc.get("video_source"),
                    author=doc.get("video_author"),
                    source_url=doc.get("video_source_url"),
                    duration_sec=doc.get("video_duration_sec"),
                ),
                segment_id=doc["segment_id"],
                kind=doc["kind"],
                segment_start_sec=doc["start_sec"],
                segment_end_sec=doc["end_sec"],
                match_start_sec=hit.match_start_sec,
                match_end_sec=hit.match_end_sec,
                matched_text=hit.matched_text,
                match_score=hit.match_score,
                score=hit.score,
                scores=hit.scores,
                text=doc.get("text"),
                caption=doc.get("caption"),
                highlight=hit.highlight,
                play_url=play_url,
                frame_url=storage.presigned_url(doc["frame_key"]) if doc.get("frame_key") else None,
            )
        )
    return SearchResponse(
        query=body.query,
        phrase=result.parsed.phrase,
        strategy=result.strategy,
        visual_query=result.visual_query,
        total=len(hits),
        hits=hits,
    )
