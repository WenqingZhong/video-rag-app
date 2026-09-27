from src.config import Settings
from src.services.understanding.intent import Intent, find_exclusions
from src.services.understanding.llm import LLMIntentParser
from src.services.understanding.rules import parse_with_rules
from src.services.understanding.service import QueryUnderstanding, Understanding


def make_query_understanding(settings: Settings) -> QueryUnderstanding:
    llm = None
    if settings.understanding_enabled:
        llm = LLMIntentParser(settings.ollama_host, settings.understanding_model, timeout=settings.understanding_timeout)
    return QueryUnderstanding(llm)


__all__ = [
    "Intent",
    "LLMIntentParser",
    "QueryUnderstanding",
    "Understanding",
    "find_exclusions",
    "make_query_understanding",
    "parse_with_rules",
]
