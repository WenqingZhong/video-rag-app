"""OpenSearch query bodies for each search strategy. Pure functions: easy to test and to print for debugging."""

from typing import Any

HIGHLIGHT = {
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"],
    "fields": {
        "text": {"number_of_fragments": 0},
        "text.stemmed": {"number_of_fragments": 0},
        "video_title": {"number_of_fragments": 0},
    },
}


def build_filters(video_id: str | None = None, kind: str | None = None, source: str | None = None) -> list[dict[str, Any]]:
    filters = []
    if video_id:
        filters.append({"term": {"video_id": video_id}})
    if kind:
        filters.append({"term": {"kind": kind}})
    if source:
        filters.append({"term": {"video_source": source}})
    return filters


def _body(must: dict[str, Any], filters: list[dict[str, Any]], size: int) -> dict[str, Any]:
    return {"size": size, "query": {"bool": {"must": [must], "filter": filters}}, "highlight": HIGHLIGHT}


def phrase_query(phrase: str, filters: list[dict[str, Any]], size: int, slop: int = 0) -> dict[str, Any]:
    """Words in this order, at most `slop` positions apart. Uses the exact field so 'is'/'the' count."""
    return _body({"match_phrase": {"text": {"query": phrase, "slop": slop}}}, filters, size)


def fuzzy_query(phrase: str, filters: list[dict[str, Any]], size: int, minimum_should_match: str = "75%") -> dict[str, Any]:
    """Last resort for misheard words: most words must match, each allowing a typo or two (fuzziness AUTO)."""
    return _body(
        {
            "match": {
                "text": {"query": phrase, "fuzziness": "AUTO", "operator": "or", "minimum_should_match": minimum_should_match}
            }
        },
        filters,
        size,
    )


def keyword_query(text: str, filters: list[dict[str, Any]], size: int) -> dict[str, Any]:
    """BM25 relevance over stemmed transcripts and video titles (English analyzer drops stop words)."""
    return _body(
        {
            "multi_match": {
                "query": text,
                # Not the exact `text` field: it keeps stop words, so "give me a clip of a dog"
                # would match any transcript containing "me", "a" or "of".
                "fields": ["text.stemmed", "video_title^2"],
                "type": "best_fields",
                "tie_breaker": 0.3,
            }
        },
        filters,
        size,
    )
