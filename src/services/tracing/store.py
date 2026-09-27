"""Saving traces to Postgres and reading them back."""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from src.db.interfaces.base import BaseDatabase
from src.models import LLMCallRecord, SpanRecord
from src.services.tracing.tracer import Span


def _plain(attributes: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy: a stray object in an attribute must not lose the whole trace."""
    return json.loads(json.dumps(attributes, default=str))


class TraceStore:
    def __init__(self, database: BaseDatabase):
        self.database = database

    def save(self, spans: list[Span]) -> None:
        """One insert per request or task, after it has finished."""
        with self.database.get_session() as session:
            session.add_all(
                SpanRecord(
                    span_id=s.span_id,
                    trace_id=s.trace_id,
                    parent_id=s.parent_id,
                    name=s.name[:128],
                    service=s.service,
                    started_at=s.started_at,
                    duration_ms=s.duration_ms if s.duration_ms is not None else 0.0,
                    status=s.status,
                    error=s.error,
                    attributes=_plain(s.attributes),
                )
                for s in spans
            )
            session.commit()


def _span_out(s: SpanRecord) -> dict:
    return {
        "span_id": s.span_id,
        "parent_id": s.parent_id,
        "name": s.name,
        "service": s.service,
        "started_at": s.started_at,
        "duration_ms": s.duration_ms,
        "status": s.status,
        "error": s.error,
        "attributes": s.attributes,
    }


def get_trace(session: Session, trace_id: str) -> dict | None:
    """All spans of one trace (API and workers), in start order, plus the tokens it spent."""
    spans = session.scalars(select(SpanRecord).where(SpanRecord.trace_id == trace_id).order_by(SpanRecord.started_at)).all()
    if not spans:
        return None
    usage = session.execute(
        select(
            func.count().label("calls"),
            func.coalesce(func.sum(LLMCallRecord.prompt_tokens + LLMCallRecord.output_tokens), 0).label("tokens"),
            func.coalesce(func.sum(LLMCallRecord.cost_usd), 0).label("cost_usd"),
        ).where(LLMCallRecord.request_id == trace_id)
    ).one()
    return {
        "trace_id": trace_id,
        "spans": [_span_out(s) for s in spans],
        "llm": {"calls": usage.calls, "tokens": int(usage.tokens), "cost_usd": round(float(usage.cost_usd), 6)},
    }


def list_traces(
    session: Session, name: str | None = None, since: datetime | None = None, slowest: bool = False, limit: int = 20
) -> list[dict]:
    """Root spans (one per request or stand-alone task): the most recent, or the slowest."""
    query = select(SpanRecord).where(SpanRecord.parent_id.is_(None))
    if name:
        query = query.where(SpanRecord.name == name)
    if since:
        query = query.where(SpanRecord.started_at >= since)
    order = SpanRecord.duration_ms.desc() if slowest else SpanRecord.started_at.desc()
    return [{"trace_id": s.trace_id, **_span_out(s)} for s in session.scalars(query.order_by(order).limit(limit))]


def delete_spans_before(session: Session, before: datetime) -> int:
    """Retention: traces are for recent debugging. Token history (llm_calls) is kept."""
    deleted = session.execute(delete(SpanRecord).where(SpanRecord.started_at < before)).rowcount
    session.commit()
    return deleted


def waterfall(trace: dict, width: int = 40) -> str:
    """A text timeline of one trace, children indented under their parents."""
    spans = trace["spans"]
    if not spans:
        return ""
    t0 = min(s["started_at"] for s in spans)
    end = max(s["started_at"].timestamp() * 1000 + s["duration_ms"] for s in spans)
    total = max(end - t0.timestamp() * 1000, 1e-6)
    children: dict[str | None, list[dict]] = {}
    ids = {s["span_id"] for s in spans}
    for s in spans:
        # a parent from another trace (or not saved) → show as a root
        children.setdefault(s["parent_id"] if s["parent_id"] in ids else None, []).append(s)

    lines = []

    def walk(parent: str | None, depth: int) -> None:
        for s in children.get(parent, []):
            offset = (s["started_at"].timestamp() * 1000 - t0.timestamp() * 1000) / total
            start = int(offset * width)
            bar = " " * start + "█" * max(1, int(s["duration_ms"] / total * width))
            label = ("  " * depth + s["name"])[:34]
            flag = " ✗" if s["status"] == "error" else ""
            lines.append(f"{label:<34} {s['service']:<11} {s['duration_ms']:>9.1f} ms |{bar:<{width}}|{flag}")
            walk(s["span_id"], depth + 1)

    walk(None, 0)
    return "\n".join(lines)
