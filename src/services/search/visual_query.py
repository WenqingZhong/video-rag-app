"""Strip request phrasing before embedding a query ("give me a clip of a dog" → "a dog").

CLIP compares the text with images; words like "give me" or "clip" describe the REQUEST, not the picture,
and only add noise. This is a small heuristic; /ask's request understanding (services/understanding) extracts what the user wants properly.
"""

import re

_REQUEST_PREFIX = re.compile(
    r"^\s*(?:(?:please|can you|could you|i want|i need|i'd like)\s+)*"
    r"(?:(?:give|show|find|get|send)(?:\s+me)?\s+)?"
    r"(?:(?:a|an|the|some)\s+)?"
    r"(?:(?:video|clip|footage|scene|shot|part|moment)s?\s+(?:of|with|where|showing)\s+)?",
    re.IGNORECASE,
)


def visual_query_text(query: str) -> str:
    cleaned = _REQUEST_PREFIX.sub("", query).strip(" ?.!")
    return cleaned or query.strip()
