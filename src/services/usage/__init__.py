from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.usage.models import WASTED, LLMCall, Operation, Outcome
from src.services.usage.pricing import Pricing
from src.services.usage.recorder import UsageRecorder


def make_pricing(settings: Settings) -> Pricing:
    return Pricing(settings.llm_price_reference, settings.llm_price_input_per_mtok, settings.llm_price_output_per_mtok)


def make_usage_recorder(settings: Settings, database: BaseDatabase, origin: str) -> UsageRecorder | None:
    if not settings.usage_tracking_enabled:
        return None
    return UsageRecorder(database, make_pricing(settings), origin)


__all__ = [
    "WASTED",
    "LLMCall",
    "Operation",
    "Outcome",
    "Pricing",
    "UsageRecorder",
    "make_pricing",
    "make_usage_recorder",
]
