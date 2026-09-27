"""Rule-based request parsing: the fallback when the LLM is off, down, slow, or its answer fails the guards."""

import re

from src.services.search.query_parser import parse_query
from src.services.search.visual_query import visual_query_text
from src.services.understanding.intent import Intent, find_exclusions, is_placeholder, strip_exclusions

_SAYS = re.compile(r"\b(?:says?|said|saying|mentions?|mentioned|goes)\s+(?:that\s+)?(?P<phrase>.+)$", re.IGNORECASE)
_TALKS_ABOUT = re.compile(
    r"\b(?:talks?|talked|talking|speaks?|discuss(?:es|ed)?|explains?)\s+(?:about\s+)?(?P<topic>.+)$", re.IGNORECASE
)


def parse_with_rules(query: str) -> Intent:
    quoted = parse_query(query).phrase
    if quoted:
        return Intent(type="quote", phrase=quoted)
    if match := _SAYS.search(query):
        return Intent(type="quote", phrase=match.group("phrase").strip(" ?.!\"'“”‘’"))
    if match := _TALKS_ABOUT.search(query):
        return Intent(type="topic", topic=match.group("topic").strip(" ?.!"))
    exclude = find_exclusions(query)
    visual = visual_query_text(strip_exclusions(query))
    if is_placeholder(visual) and exclude:  # only what to avoid: no subject to search for
        visual = None
    return Intent(type="visual", visual=visual or (None if exclude else query.strip()), exclude=exclude)
