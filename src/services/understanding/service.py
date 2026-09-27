"""Request → Intent. The LLM first; its answer is checked against the request by guards; rules as fallback."""

import logging
import re
import time
from dataclasses import dataclass, field

import httpx
from pydantic import ValidationError

from src.services.understanding.intent import Intent, find_exclusions, is_placeholder
from src.services.understanding.llm import LLMIntentParser
from src.services.understanding.rules import parse_with_rules

logger = logging.getLogger(__name__)

# Share of the search text's words that must appear in the request (quotes must be nearly verbatim).
MIN_GROUNDED = {"quote": 0.8, "topic": 0.5, "visual": 0.5}

# quote/topic mean "something SAID". Without a speech word or quote marks, a request is about what is SHOWN.
_SPEECH_CUE = re.compile(
    r"\b(?:say|says|said|saying|mention\w*|talk\w*|discuss\w*|speak\w*|spoke|explain\w*|line|quote\w*|words?|tells?|told|"
    r"goes|asks?|asked)\b|[\"“”‘’]|(?<!\w)'|'(?!\w)",
    re.IGNORECASE,
)


@dataclass
class Understanding:
    intent: Intent
    source: str  # "llm" | "llm+rules" (rules added exclusions) | "rules" (fallback)
    seconds: float
    llm_raw: Intent | None = None  # what the LLM answered, before the guards (for debugging / the demo)
    notes: list[str] = field(default_factory=list)  # what the guards changed, and why


def _words(text: str) -> list[str]:
    # Quote marks must not stick to words: a closing ’ would turn "everything" into "everything'".
    tokens = re.findall(r"[a-z0-9']+", text.lower().replace("’", "'").replace("‘", "'"))
    return [t.strip("'") for t in tokens if t.strip("'")]


_FUNCTION_WORDS = {
    "a", "an", "the", "of", "on", "in", "at", "to", "for", "with", "and", "or", "is", "are", "some", "any", "me",
}  # fmt: skip


def _grounded(text: str, query: str, minimum: float) -> bool:
    """Did the user actually say (most of) these words? Catches invented phrases and added details.

    Only meaningful words count: in "a dog sitting on a couch", "a"/"on" would match any request.
    (If the text is nothing but small words, e.g. a quote like "it is what it is", all words count.)
    """
    words, said = _words(text), set(_words(query))
    words = [w for w in words if w not in _FUNCTION_WORDS] or words
    if not words:
        return False
    stems = {w.rstrip("s") for w in said}
    return sum(1 for w in words if w in said or w.rstrip("s") in stems) / len(words) >= minimum


def _merge_exclusions(candidates: list[str], query: str) -> list[str]:
    """Only words the user said; overlapping terms ("siamese" / "siamese cats") collapse to the longer one."""
    grounded = [c.strip().lower() for c in candidates if c.strip() and _grounded(c, query, 1.0)]
    merged: list[str] = []
    for term in sorted(dict.fromkeys(grounded), key=len, reverse=True):
        if not any(term in kept for kept in merged):
            merged.append(term)
    return merged


class QueryUnderstanding:
    def __init__(self, llm: LLMIntentParser | None):
        self.llm = llm

    def understand(self, query: str) -> Understanding:
        started = time.perf_counter()

        def done(intent: Intent, source: str, raw: Intent | None, notes: list[str]) -> Understanding:
            return Understanding(intent, source, round(time.perf_counter() - started, 3), raw, notes)

        if self.llm is None:
            return done(parse_with_rules(query), "rules", None, ["LLM disabled"])
        try:
            raw = self.llm.parse(query)
        except (httpx.HTTPError, ValidationError, ValueError, KeyError) as exc:
            logger.warning("LLM understanding failed, using rules: %s", exc)
            return done(parse_with_rules(query), "rules", None, [f"LLM error: {exc}"])

        notes: list[str] = []
        intent_type = raw.type
        # Guard 1 (type): quote/topic need a sign of speech; bare nouns like "canine" are visual requests.
        if intent_type in ("quote", "topic") and not _SPEECH_CUE.search(query):
            notes.append(f"type {intent_type} → visual: no speech word or quote marks in the request")
            intent_type = "visual"
        # Guard 2 (fields): keep only the field for the type; the model fills the others with guesses.
        text = raw.text
        field_name = {"quote": "phrase", "topic": "topic", "visual": "visual"}[intent_type]
        # "No subject" (visual null, or just "anything") is a valid answer: Phase 3 asks the user instead of searching.
        no_subject = intent_type == "visual" and is_placeholder(text)
        if no_subject:
            text = None
            notes.append("no subject: nothing to search for (only what to avoid, or a placeholder like 'something')")
        # Guard 3 (grounding): the search text must use the user's own words, or the rules take over.
        elif not _grounded(text, query, MIN_GROUNDED[intent_type]):
            fallback = parse_with_rules(query)
            notes.append(f"{intent_type} text {text!r} not in the request's words → rules")
            return done(fallback, "rules", raw, notes)
        # Guard 4 (exclusions): union with the rules (the LLM drops "no X"), only words the user said.
        exclude = _merge_exclusions([*raw.exclude, *find_exclusions(query)], query)
        added = [e for e in exclude if e not in [x.lower() for x in raw.exclude]]
        if added:
            notes.append(f"exclusions added by rules: {added}")
        intent = Intent(type=intent_type, exclude=exclude, **{field_name: text})
        return done(intent, "llm+rules" if added else "llm", raw, notes)
