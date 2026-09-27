from src.config import Settings
from src.services.cache import CacheClient
from src.services.understanding.cache import UnderstandingCache
from src.services.understanding.intent import Intent, find_exclusions
from src.services.understanding.llm import LLMIntentParser
from src.services.understanding.rules import parse_with_rules
from src.services.understanding.service import QueryUnderstanding, Understanding
from src.services.usage import UsageRecorder


def make_query_understanding(
    settings: Settings, recorder: UsageRecorder | None = None, cache: CacheClient | None = None
) -> QueryUnderstanding:
    llm = None
    if settings.understanding_enabled:
        llm = LLMIntentParser(settings.ollama_host, settings.understanding_model, timeout=settings.understanding_timeout)
    understanding_cache = None
    if cache is not None and settings.cache_enabled:
        understanding_cache = UnderstandingCache(cache, settings.understanding_model, settings.understanding_cache_ttl_sec)
    return QueryUnderstanding(llm, recorder, understanding_cache)


__all__ = [
    "Intent",
    "LLMIntentParser",
    "QueryUnderstanding",
    "Understanding",
    "UnderstandingCache",
    "find_exclusions",
    "make_query_understanding",
    "parse_with_rules",
]
