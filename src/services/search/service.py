"""Search over segments: quote → phrase cascade (exact → slop → fuzzy); otherwise keyword (BM25)."""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import Settings
from src.services.opensearch import OpenSearchService
from src.services.search import query_builder as qb
from src.services.search.matching import locate_phrase
from src.services.search.query_parser import ParsedQuery, parse_query

logger = logging.getLogger(__name__)

OVERFETCH = 3  # fetch extra hits because overlapping windows collapse into one moment


@dataclass
class SearchHit:
    source: dict[str, Any]  # the OpenSearch document
    score: float
    match_start_sec: float
    match_end_sec: float
    match_score: float | None  # how well the quote aligned with the words (1.0 = exact); None for keyword hits
    matched_text: str | None
    highlight: str | None


@dataclass
class SearchResult:
    parsed: ParsedQuery
    strategy: str  # exact_phrase | phrase_with_slop | fuzzy_phrase | keyword | none
    hits: list[SearchHit] = field(default_factory=list)


def _overlaps(a: SearchHit, b: SearchHit) -> bool:
    return (
        a.source["video_id"] == b.source["video_id"]
        and a.match_start_sec < b.match_end_sec
        and b.match_start_sec < a.match_end_sec
    )


class SearchService:
    def __init__(self, opensearch: OpenSearchService, settings: Settings):
        self.opensearch = opensearch
        self.settings = settings

    def _strategies(self, parsed: ParsedQuery, filters: list[dict], size: int) -> list[tuple[str, dict]]:
        if not parsed.phrase:
            return [("keyword", qb.keyword_query(parsed.text, filters, size))]
        phrase = parsed.phrase
        return [
            ("exact_phrase", qb.phrase_query(phrase, filters, size, slop=0)),
            ("phrase_with_slop", qb.phrase_query(phrase, filters, size, slop=self.settings.search_phrase_slop)),
            ("fuzzy_phrase", qb.fuzzy_query(phrase, filters, size, self.settings.search_fuzzy_min_match)),
        ]

    def search(
        self,
        query: str,
        video_id: str | None = None,
        kind: str | None = None,
        source: str | None = None,
        size: int = 10,
        group_by_video: bool = False,
    ) -> SearchResult:
        parsed = parse_query(query)
        # A quote is something *said*: search speech unless the caller asked for a specific kind.
        effective_kind = kind or ("speech" if parsed.phrase else None)
        filters = qb.build_filters(video_id=video_id, kind=effective_kind, source=source)
        fetch = min(size * OVERFETCH, self.settings.search_max_size * OVERFETCH)

        for strategy, body in self._strategies(parsed, filters, fetch):
            raw_hits = self.opensearch.search(body)["hits"]["hits"]
            hits = [h for h in (self._to_hit(raw, parsed) for raw in raw_hits) if h is not None]
            if hits:
                logger.info("search %r → %s (%s hits)", query, strategy, len(hits))
                return SearchResult(parsed=parsed, strategy=strategy, hits=self._dedupe(hits, size, group_by_video))
        return SearchResult(parsed=parsed, strategy="none")

    def _to_hit(self, raw: dict[str, Any], parsed: ParsedQuery) -> SearchHit | None:
        doc = raw["_source"]
        highlight = next(iter(raw.get("highlight", {}).values()), [None])[0]
        start, end, match_score, matched_text = doc["start_sec"], doc["end_sec"], None, None
        if parsed.phrase:
            match = locate_phrase(parsed.phrase, doc.get("words") or [])
            if match is None:  # OpenSearch matched loosely but the words don't really contain the phrase
                return None
            start, end, match_score, matched_text = match.start_sec, match.end_sec, match.score, match.text(doc["words"])
        return SearchHit(
            source=doc,
            score=float(raw["_score"] or 0.0),
            match_start_sec=start,
            match_end_sec=end,
            match_score=match_score,
            matched_text=matched_text,
            highlight=highlight,
        )

    @staticmethod
    def _dedupe(hits: list[SearchHit], size: int, group_by_video: bool) -> list[SearchHit]:
        """Overlapping windows repeat the same moment: keep the best-aligned, highest-scoring copy."""
        ranked = sorted(hits, key=lambda h: (h.match_score or 0.0, h.score), reverse=True)
        kept: list[SearchHit] = []
        seen_videos: set[str] = set()
        for hit in ranked:
            if any(_overlaps(hit, k) for k in kept):
                continue
            if group_by_video and hit.source["video_id"] in seen_videos:
                continue
            kept.append(hit)
            seen_videos.add(hit.source["video_id"])
            if len(kept) == size:
                break
        return kept
