from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, status

from src.dependencies import SessionDep
from src.schemas.api.traces import TraceOut, TraceSummary
from src.services.tracing import get_trace, list_traces, waterfall

router = APIRouter(prefix="/traces", tags=["Traces"])


@router.get("", response_model=list[TraceSummary])
def recent_traces(
    session: SessionDep,
    name: str | None = Query(None, description='Root span name, e.g. "POST /api/v1/ask"'),
    hours: float | None = Query(None, gt=0, description="Only the last N hours"),
    slowest: bool = Query(False, description="Slowest first instead of most recent first"),
    limit: int = Query(20, ge=1, le=200),
) -> list[TraceSummary]:
    """One line per request or stand-alone task (its root span)."""
    since = datetime.now(UTC) - timedelta(hours=hours) if hours else None
    return [TraceSummary(**t) for t in list_traces(session, name=name, since=since, slowest=slowest, limit=limit)]


@router.get("/{trace_id}", response_model=TraceOut)
def trace(trace_id: str, session: SessionDep) -> TraceOut:
    """Every span of one trace, across the API and the workers. The id is the X-Request-ID response header."""
    found = get_trace(session, trace_id)
    if found is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace not found (or not saved yet)")
    return TraceOut(**found, waterfall=waterfall(found))
