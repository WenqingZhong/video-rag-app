"""What the user wants, as structured search instructions, plus rule-based negation spotting."""

import re
from typing import Literal

from pydantic import BaseModel, Field

IntentType = Literal["quote", "topic", "visual"]


class Intent(BaseModel):
    type: IntentType = Field(..., description="quote: words someone SAYS · topic: something DISCUSSED · visual: something SHOWN")
    phrase: str | None = Field(None, description="quote: the exact words")
    topic: str | None = Field(None, description="topic: the subject")
    visual: str | None = Field(None, description="visual: what should be visible")
    exclude: list[str] = Field(default_factory=list, description="things that must NOT appear")

    @property
    def has_subject(self) -> bool:
        """False when the user said only what to avoid ("exclude people", "anything without cars"): ask, don't search."""
        return not is_placeholder(self.text)

    @property
    def text(self) -> str:
        """The search text for this intent's type."""
        return {"quote": self.phrase, "topic": self.topic, "visual": self.visual}[self.type] or ""


# Words that stand in for "whatever": searching for them is meaningless.
_PLACEHOLDERS = {
    "anything", "something", "everything", "stuff", "things", "any", "some", "footage", "video", "videos", "clip",
    "clips", "exclude", "excluding", "nothing", "whatever", "one", "ones",
    # chit-chat: a greeting or thanks is not a request for footage
    "hi", "hello", "hey", "there", "thanks", "thank", "you", "ok", "okay", "yo", "test", "testing",
}  # fmt: skip


def is_placeholder(text: str | None) -> bool:
    words = re.findall(r"[a-z']+", (text or "").lower())
    filler = {"a", "an", "the", "me", "please", "with", "in", "it", "i", "we", "just"}
    return all(w in _PLACEHOLDERS or w in filler for w in words)


# "no people", "without cars or trucks", "not at night", "but not the city", "excluding boats"
_NEGATION = re.compile(
    r"\b(?:but\s+)?(?:no|without|not|exclude|excluding|except|nothing\s+with|(?:don'?t|do\s+not)\s+want)\s+(?P<clause>[^,.;!?]*?)(?=\s+but\b|[,.;!?]|$)",
    re.IGNORECASE,
)
_LEADING = re.compile(r"^(?:any|a|an|the|at|in|on|during|some|no|not|without)\s+", re.IGNORECASE)
_POLITE = re.compile(r"\s+(?:please|thanks|thank you|pls)$", re.IGNORECASE)
_DANGLING = re.compile(r"\s+(?:with|and|or|but|in|on|at)$", re.IGNORECASE)


def find_exclusions(query: str) -> list[str]:
    """Rule-based negation spotting. A safety net: the LLM sometimes drops "no X" entirely."""
    found: list[str] = []
    for match in _NEGATION.finditer(query):
        for part in re.split(r"\s+(?:or|and|nor)\s+|\s*/\s*", match.group("clause")):
            item = _POLITE.sub("", part.strip().lower())
            while (stripped := _LEADING.sub("", item)) != item:
                item = stripped
            if item:
                found.append(item)
    return list(dict.fromkeys(found))


def strip_exclusions(text: str) -> str:
    """Remove negated clauses from search text ("a beach with no people" → "a beach")."""
    cleaned = re.sub(r"\s{2,}", " ", _NEGATION.sub("", text)).strip(" ,.;!?")
    while (stripped := _DANGLING.sub("", cleaned)) != cleaned:
        cleaned = stripped
    return cleaned
