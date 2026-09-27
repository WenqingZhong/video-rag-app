"""Token and cost questions, answered from the llm_calls ledger."""

from datetime import datetime

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from src.models import LLMCallRecord as C
from src.models import Video
from src.services.usage.models import WASTED

_TOTALS = (
    func.count().label("calls"),
    func.coalesce(func.sum(C.prompt_tokens), 0).label("prompt_tokens"),
    func.coalesce(func.sum(C.output_tokens), 0).label("output_tokens"),
    func.coalesce(func.sum(C.cost_usd), 0).label("cost_usd"),
)


def _totals(row) -> dict:
    prompt, output = int(row.prompt_tokens), int(row.output_tokens)
    return {
        "calls": row.calls,
        "prompt_tokens": prompt,
        "output_tokens": output,
        "total_tokens": prompt + output,
        "cost_usd": round(float(row.cost_usd), 6),
    }


def usage_report(session: Session, since: datetime | None = None, origin: str | None = None) -> dict:
    def scoped(query: Select, calls_only: bool = True) -> Select:
        if calls_only:  # cache hits spent nothing: they have their own section below
            query = query.where(C.outcome != "cache_hit")
        if since is not None:
            query = query.where(C.created_at >= since)
        if origin is not None:
            query = query.where(C.origin == origin)
        return query

    def grouped(*columns) -> list:
        return list(session.execute(scoped(select(*columns, *_TOTALS).group_by(*columns)).order_by(*columns)))

    overall = _totals(session.execute(scoped(select(*_TOTALS))).one())
    by_operation = {r.operation: _totals(r) for r in grouped(C.operation)}
    for totals in by_operation.values():
        totals["tokens_per_call"] = round(totals["total_tokens"] / totals["calls"], 1)
        totals["cost_per_1000_calls_usd"] = round(1000 * totals["cost_usd"] / totals["calls"], 4)

    outcomes = {r.outcome: _totals(r) for r in grouped(C.outcome) if r.outcome != "ok"}
    understand_tokens = by_operation.get("understand", {}).get("total_tokens", 0)
    wasted = sum(t["total_tokens"] for o, t in outcomes.items() if o in WASTED)

    speed = session.execute(
        scoped(
            select(
                C.operation,
                func.avg(C.prompt_sec).label("prompt_sec"),
                func.avg(C.output_sec).label("output_sec"),
                func.sum(C.output_tokens).label("output_tokens"),
                func.sum(C.output_sec).label("output_sec_total"),
            ).group_by(C.operation)
        )
    )
    per_video = session.execute(
        scoped(
            select(
                C.video_id,
                Video.title,
                Video.duration_sec,
                *_TOTALS,
            )
            .join(Video, Video.id == C.video_id, isouter=True)
            .where(C.operation == "caption", C.video_id.is_not(None))
            .group_by(C.video_id, Video.title, Video.duration_sec)
            .order_by(func.sum(C.prompt_tokens + C.output_tokens).desc())
        )
    )

    videos = []
    for r in per_video:
        totals = _totals(r)
        minutes = (r.duration_sec or 0) / 60
        totals["tokens_per_minute"] = round(totals["total_tokens"] / minutes) if minutes else None
        videos.append({"video_id": r.video_id, "title": r.title, "keyframes": r.calls, **totals})

    hits = session.execute(
        scoped(
            select(
                C.operation,
                func.count().label("hits"),
                func.coalesce(func.sum(C.saved_tokens), 0).label("saved_tokens"),
                func.coalesce(func.sum(C.saved_cost_usd), 0).label("saved_cost_usd"),
            )
            .where(C.outcome == "cache_hit")
            .group_by(C.operation),
            calls_only=False,
        )
    )
    cache = {}
    for r in hits:
        calls = by_operation.get(r.operation, {}).get("calls", 0)
        cache[r.operation] = {
            "hits": r.hits,
            "hit_rate": round(r.hits / (r.hits + calls), 3),  # of all requests that needed this step
            "saved_tokens": int(r.saved_tokens),
            "saved_cost_usd": round(float(r.saved_cost_usd), 6),
        }

    return {
        "overall": overall,
        "cache": cache,
        "by_operation": by_operation,
        "by_origin": {r.origin: _totals(r) for r in grouped(C.origin)},
        "understand_outcomes": outcomes,
        "understand_wasted_share": round(wasted / understand_tokens, 3) if understand_tokens else 0.0,
        "speed": {
            r.operation: {
                "avg_prompt_sec": round(r.prompt_sec or 0, 3),
                "avg_output_sec": round(r.output_sec or 0, 3),
                "output_tokens_per_sec": round(r.output_tokens / r.output_sec_total, 1) if r.output_sec_total else None,
            }
            for r in speed
        },
        "videos": videos,
    }
