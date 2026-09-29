"""Request → Intent. The LLM first; its answer is checked against the request by guards; rules as fallback."""

import logging
import re
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.services.llm import ModelError
from src.services.tracing import span
from src.services.understanding.cache import UnderstandingCache
from src.services.understanding.intent import Intent, find_exclusions, is_placeholder
from src.services.understanding.llm import InvalidReply, LLMIntentParser
from src.services.understanding.rules import parse_with_rules
from src.services.usage import LLMCall, Outcome, UsageRecorder

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
    llm_call: LLMCall | None = None  # tokens and time the model spent (None: rules only, or the call failed)
    llm_outcome: Outcome | None = None  # what became of the model's answer (see src/services/usage/models.py)
    cost_usd: float = 0.0  # estimated, at the configured hosted price
    cached: bool = False  # served from the understanding cache: no model call was made
    avoided_call: LLMCall | None = None  # on a cache hit: the call that was not made (its tokens were saved)
    saved_cost_usd: float = 0.0


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


def _llm_attributes(target, call: LLMCall) -> None:
    target.set(
        prompt_tokens=call.prompt_tokens,
        output_tokens=call.output_tokens,
        load_sec=call.load_sec,
        prompt_sec=call.prompt_sec,
        output_sec=call.output_sec,
    )


def _cache_state(understanding: "QueryUnderstanding", result: Understanding) -> str:
    if understanding.cache is None or understanding.llm is None:
        return "off"
    return "hit" if result.cached else "miss"


def _outcome(source: str, raw: Intent, intent: Intent) -> Outcome:
    if source == "rules":
        return "rejected"
    same = raw.type == intent.type and raw.text == intent.text and sorted(raw.exclude) == sorted(intent.exclude)
    return "accepted" if same else "adjusted"


class QueryUnderstanding:
    def __init__(
        self,
        llm: LLMIntentParser | None,
        recorder: UsageRecorder | None = None,
        cache: "UnderstandingCache | None" = None,
    ):
        self.llm = llm
        self.recorder = recorder
        self.cache = cache  # None: always ask the model

    def understand(self, query: str, request_id: str | None = None) -> Understanding:
        with span("understand") as step:
            result = self._from_cache(query, request_id)
            if result is None:
                result = self._understand(query, request_id)
                # Only answers the model actually gave: a failed call (→ rules) should be retried next time.
                if self.cache is not None and result.llm_call is not None:
                    self.cache.put(query, result)
            step.set(
                source=result.source, outcome=result.llm_outcome, intent_type=result.intent.type, cache=_cache_state(self, result)
            )
            if result.llm_call is not None:
                step.set(tokens=result.llm_call.total_tokens, cost_usd=result.cost_usd)
            if result.avoided_call is not None:
                step.set(saved_tokens=result.avoided_call.total_tokens)
            return result

    def _from_cache(self, query: str, request_id: str | None) -> Understanding | None:
        if self.cache is None or self.llm is None:
            return None
        started = time.perf_counter()
        hit = self.cache.get(query)
        if hit is None:
            return None
        avoided = hit.pop("avoided")
        result = Understanding(seconds=round(time.perf_counter() - started, 3), cached=True, avoided_call=avoided, **hit)
        if self.recorder is not None:
            result.saved_cost_usd = self.recorder.record_saved(avoided, request_id=request_id)
        return result

    def _understand(self, query: str, request_id: str | None) -> Understanding:
        started = time.perf_counter()
        call: LLMCall | None = None

        def done(intent: Intent, source: str, raw: Intent | None, notes: list[str]) -> Understanding:
            result = Understanding(intent, source, round(time.perf_counter() - started, 3), raw, notes, call)
            if call is not None:
                result.llm_outcome = "invalid" if raw is None else _outcome(source, raw, intent)
                if self.recorder is not None:
                    result.cost_usd = self.recorder.record(call, result.llm_outcome, request_id=request_id)
            return result

        if self.llm is None:
            return done(parse_with_rules(query), "rules", None, ["LLM disabled"])
        with span("llm.understand", model=self.llm.model) as llm_span:
            try:
                raw, call = self.llm.parse(query)
            except InvalidReply as exc:
                call = exc.call
                llm_span.fail(exc)
                logger.warning("LLM reply invalid, using rules: %s", exc)
                failure = f"LLM reply invalid: {exc}"
            except (ModelError, ValidationError, ValueError, KeyError) as exc:
                llm_span.fail(exc)
                logger.warning("LLM understanding failed, using rules: %s", exc)
                failure = f"LLM error: {exc}"
            else:
                failure = None
            if call is not None:
                _llm_attributes(llm_span, call)
        if failure is not None:
            return done(parse_with_rules(query), "rules", None, [failure])

        notes: list[str] = []
        intent_type = raw.type
        # Guard 1 (type): quote/topic need a sign of speech; bare nouns like "canine" are visual requests.
        if intent_type in ("quote", "topic") and not _SPEECH_CUE.search(query):
            notes.append(f"type {intent_type} → visual: no speech word or quote marks in the request")
            intent_type = "visual"
        # Guard 2 (fields): keep only the field for the type; the model fills the others with guesses.
        text = raw.text
        field_name = {"quote": "phrase", "topic": "topic", "visual": "visual"}[intent_type]
        # "No subject" (visual null, or just "anything") is a valid answer: /ask then asks the user instead of searching.
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
