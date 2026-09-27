"""Writes each model call to the llm_calls ledger. Never breaks the request or task it measures."""

import logging

from sqlalchemy.exc import SQLAlchemyError

from src.db.interfaces.base import BaseDatabase
from src.models.usage import LLMCallRecord
from src.services.tracing import current_trace_id
from src.services.usage.models import LLMCall, Outcome
from src.services.usage.pricing import Pricing

logger = logging.getLogger(__name__)


class UsageRecorder:
    def __init__(self, database: BaseDatabase, pricing: Pricing, origin: str):
        self.database = database
        self.pricing = pricing
        self.origin = origin  # "api" | "worker" | "eval": keeps evaluation runs apart from real traffic

    def record_saved(self, avoided: LLMCall, *, request_id: str | None = None) -> float:
        """A cache hit: the call `avoided` was not made. Returns the estimated cost saved."""
        saved = self.pricing.cost(avoided)
        empty = LLMCall(avoided.operation, avoided.model, prompt_tokens=0, output_tokens=0)
        self._store(empty, "cache_hit", None, request_id, saved_tokens=avoided.total_tokens, saved_cost_usd=saved)
        return saved

    def record(self, call: LLMCall, outcome: Outcome, *, video_id: str | None = None, request_id: str | None = None) -> float:
        """Store the call and return its estimated cost. A database outage loses the row, not the request."""
        cost = self.pricing.cost(call)
        self._store(call, outcome, video_id, request_id, cost_usd=cost)
        return cost

    def _store(
        self,
        call: LLMCall,
        outcome: Outcome,
        video_id: str | None,
        request_id: str | None,
        cost_usd: float = 0.0,
        saved_tokens: int = 0,
        saved_cost_usd: float = 0.0,
    ) -> None:
        row = LLMCallRecord(
            operation=call.operation,
            outcome=outcome,
            origin=self.origin,
            model=call.model,
            prompt_tokens=call.prompt_tokens,
            output_tokens=call.output_tokens,
            images=call.images,
            load_sec=call.load_sec,
            prompt_sec=call.prompt_sec,
            output_sec=call.output_sec,
            total_sec=call.total_sec,
            cost_usd=cost_usd,
            price_reference=self.pricing.reference,
            saved_tokens=saved_tokens,
            saved_cost_usd=saved_cost_usd,
            video_id=video_id,
            request_id=request_id or current_trace_id(),  # the trace this call belongs to
        )
        try:
            with self.database.get_session() as session:
                session.add(row)
                session.commit()
        except (SQLAlchemyError, RuntimeError) as exc:  # RuntimeError: database never started
            logger.warning("usage not recorded (%s %s tokens): %s", call.operation, call.total_tokens, exc)
