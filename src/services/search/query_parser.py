"""Find the quoted phrase in a natural-language request, with plain rules (the LLM-based version: services/understanding)."""

import re
from dataclasses import dataclass

# Lazy match up to a closing quote that is NOT followed by a letter, so apostrophes inside the quote
# ("‘I don’t know’", "'we're here'") don't end it early. Curly ’ doubles as the typographic apostrophe.
_QUOTE_PATTERNS = [
    re.compile(r"“(.+?)”"),
    re.compile(r'"(.+?)"'),
    re.compile(r"‘(.+?)’(?!\w)"),
    re.compile(r"(?<!\w)'(.+?)'(?!\w)"),
]


@dataclass(frozen=True)
class ParsedQuery:
    raw: str
    phrase: str | None  # quoted text → phrase search over transcripts
    text: str  # the full query → keyword search when there is no phrase


def parse_query(query: str) -> ParsedQuery:
    query = query.strip()
    matches = [m for pattern in _QUOTE_PATTERNS for m in pattern.finditer(query)]
    if matches:
        first = min(matches, key=lambda m: m.start())
        phrase = first.group(1).strip(" .,!?;:")
        if re.search(r"\w", phrase):
            return ParsedQuery(raw=query, phrase=phrase, text=query)
    return ParsedQuery(raw=query, phrase=None, text=query)
