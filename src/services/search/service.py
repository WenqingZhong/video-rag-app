"""Search over segments.

- quoted phrase → phrase cascade over transcripts (exact → slop → fuzzy), with word-level times
- otherwise     → hybrid: keyword (BM25) + vector (CLIP kNN over keyframes), fused with RRF
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from src.config import Settings
from src.services.embeddings import EmbeddingClient
from src.services.opensearch import OpenSearchService
from src.services.search import query_builder as qb
from src.services.search.fusion import reciprocal_rank_fusion
from src.services.search.matching import locate_phrase
from src.services.search.query_parser import ParsedQuery, parse_query
from src.services.search.visual_query import visual_query_text
from src.services.understanding.intent import Intent

logger = logging.getLogger(__name__)

OVERFETCH = 3  # fetch extra hits because overlapping windows collapse into one moment
SearchMode = Literal["auto", "hybrid", "keyword", "vector"]


class SearchUnavailable(RuntimeError):
    """A requested retriever (e.g. vector search) can't run right now."""


@dataclass
class SearchHit:
    source: dict[str, Any]  # the OpenSearch document
    score: float  # BM25 for keyword/phrase results; RRF for hybrid; similarity for vector-only
    match_start_sec: float
    match_end_sec: float
    match_score: float | None  # how well the quote aligned with the words (1.0 = exact); None otherwise
    matched_text: str | None
    highlight: str | None
    scores: dict[str, float] = field(default_factory=dict)  # per-retriever scores and ranks (hybrid)


@dataclass
class SearchResult:
    parsed: ParsedQuery
    strategy: str  # exact_phrase | phrase_with_slop | fuzzy_phrase | keyword | vector | hybrid | none
    hits: list[SearchHit] = field(default_factory=list)
    visual_query: str | None = None  # the text actually embedded for vector search


def _overlaps(a: SearchHit, b: SearchHit) -> bool:
    return (
        a.source["video_id"] == b.source["video_id"]
        and a.match_start_sec < b.match_end_sec
        and b.match_start_sec < a.match_end_sec
    )


def _mentions(hit: SearchHit, term: str) -> bool:
    """Does the hit's caption, transcript or title mention `term`? Word-prefix match: 'night' also catches 'nighttime'."""
    haystack = " ".join(str(hit.source.get(f) or "") for f in ("caption", "text", "video_title")).lower()
    words = [re.escape(w.rstrip("s")) for w in term.lower().split()]
    return bool(words) and re.search(r"\b" + r"\w*\s+".join(words), haystack) is not None


def _cosine(opensearch_score: float) -> float:
    """OpenSearch reports cosinesimil kNN scores as (1 + cosine) / 2; convert back to cosine."""
    return 2 * opensearch_score - 1


class SearchService:
    def __init__(self, opensearch: OpenSearchService, settings: Settings, embedder: EmbeddingClient | None = None):
        self.opensearch = opensearch
        self.settings = settings
        self.embedder = embedder

    def search(
        self,
        query: str,
        video_id: str | None = None,
        kind: str | None = None,
        source: str | None = None,
        size: int = 10,
        group_by_video: bool = False,
        mode: SearchMode = "auto",
    ) -> SearchResult:
        parsed = parse_query(query)
        fetch = min(size * OVERFETCH, self.settings.search_max_size * OVERFETCH)
        if parsed.phrase and mode == "auto":
            return self._phrase_search(parsed, video_id, kind, source, size, fetch, group_by_video)
        return self._semantic_search(parsed, video_id, kind, source, size, fetch, group_by_video, mode)

    def search_intent(
        self,
        intent: Intent,
        video_id: str | None = None,
        source: str | None = None,
        size: int = 10,
        group_by_video: bool = False,
    ) -> SearchResult:
        """Search driven by an understood request (Week 5) instead of raw text.

        quote  → phrase cascade over speech (exact word times)
        topic  → keyword search over speech
        visual → hybrid search over keyframes only (a transcript saying "dog" isn't footage of a dog)
        Exclusions drop hits whose caption/transcript/title mentions them. A quote/topic search that finds
        nothing is retried as visual ("three people talking" may be footage, not speech).
        """
        fetch = min(size * OVERFETCH, self.settings.search_max_size * OVERFETCH)
        text, exclude = intent.text, intent.exclude
        if intent.type == "quote":
            parsed = ParsedQuery(raw=text, phrase=text, text=text)
            result = self._phrase_search(parsed, video_id, "speech", source, size, fetch, group_by_video, exclude)
        elif intent.type == "topic":
            parsed = ParsedQuery(raw=text, phrase=None, text=text)
            result = self._semantic_search(parsed, video_id, "speech", source, size, fetch, group_by_video, "keyword", exclude)
        else:
            parsed = ParsedQuery(raw=text, phrase=None, text=text)
            return self._semantic_search(parsed, video_id, "visual", source, size, fetch, group_by_video, "auto", exclude)
        if result.hits:
            return result
        visual = ParsedQuery(raw=text, phrase=None, text=text)
        fallback = self._semantic_search(visual, video_id, "visual", source, size, fetch, group_by_video, "auto", exclude)
        if fallback.hits:
            fallback.strategy = f"{intent.type}_none→visual_{fallback.strategy}"
            return fallback
        return result

    # ---- quotes ----------------------------------------------------------------------------------------
    def _phrase_search(self, parsed, video_id, kind, source, size, fetch, group_by_video, exclude=()) -> SearchResult:
        # A quote is something *said*: search speech unless the caller asked for a specific kind.
        filters = qb.build_filters(video_id=video_id, kind=kind or "speech", source=source)
        phrase = parsed.phrase
        strategies = [
            ("exact_phrase", qb.phrase_query(phrase, filters, fetch, slop=0)),
            ("phrase_with_slop", qb.phrase_query(phrase, filters, fetch, slop=self.settings.search_phrase_slop)),
            ("fuzzy_phrase", qb.fuzzy_query(phrase, filters, fetch, self.settings.search_fuzzy_min_match)),
        ]
        for strategy, body in strategies:
            raw_hits = self.opensearch.search(body)["hits"]["hits"]
            hits = [h for h in (self._to_hit(raw, parsed) for raw in raw_hits) if h is not None]
            hits = [h for h in hits if not any(_mentions(h, term) for term in exclude)]
            if hits:
                logger.info("search %r → %s (%s hits)", parsed.raw, strategy, len(hits))
                return SearchResult(parsed=parsed, strategy=strategy, hits=self._dedupe(hits, size, group_by_video))
        return SearchResult(parsed=parsed, strategy="none")

    # ---- keyword + vector ------------------------------------------------------------------------------
    def _semantic_search(self, parsed, video_id, kind, source, size, fetch, group_by_video, mode, exclude=()) -> SearchResult:
        # Explicit keyword/vector/hybrid mode treats quotes as plain text (no phrase pinpointing).
        parsed = ParsedQuery(raw=parsed.raw, phrase=None, text=parsed.text)
        filters = qb.build_filters(video_id=video_id, kind=kind, source=source)
        use_keyword = mode in ("auto", "hybrid", "keyword")
        # Only keyframes have vectors: a speech-only search can't use the vector retriever.
        use_vector = mode in ("auto", "hybrid", "vector") and kind != "speech"

        keyword_hits: list[SearchHit] = []
        if use_keyword:
            raw = self.opensearch.search(qb.keyword_query(parsed.text, filters, fetch))["hits"]["hits"]
            keyword_hits = [self._to_hit(r, parsed) for r in raw]

        vector_hits: list[SearchHit] = []
        visual_text = None
        if use_vector:
            visual_text = visual_query_text(parsed.text)
            try:
                vector_hits = self._vector_hits(visual_text, filters, fetch, parsed)
            except (httpx.HTTPError, SearchUnavailable) as exc:
                if mode == "auto":  # degrade gracefully: keyword results are still useful
                    logger.warning("vector search unavailable, falling back to keyword: %s", exc)
                    use_vector = False
                else:
                    raise SearchUnavailable(f"vector search unavailable: {exc}") from exc

        if use_keyword and use_vector:
            strategy, hits = "hybrid", self._fuse(keyword_hits, vector_hits)
        elif use_vector:
            strategy, hits = "vector", vector_hits
        else:
            strategy, hits = "keyword", keyword_hits

        if exclude:
            hits = [h for h in hits if not any(_mentions(h, term) for term in exclude)]
        logger.info("search %r → %s (%s keyword, %s vector)", parsed.raw, strategy, len(keyword_hits), len(vector_hits))
        if not hits:
            return SearchResult(parsed=parsed, strategy="none", visual_query=visual_text)
        return SearchResult(
            parsed=parsed, strategy=strategy, hits=self._dedupe(hits, size, group_by_video), visual_query=visual_text
        )

    def _vector_hits(self, text: str, filters: list[dict], fetch: int, parsed: ParsedQuery) -> list[SearchHit]:
        if self.embedder is None:
            raise SearchUnavailable("no embedding service configured")
        vector = self.embedder.embed_texts([text])[0]
        raw = self.opensearch.search(qb.vector_query(vector, filters, fetch))["hits"]["hits"]
        hits = []
        for r in raw:
            similarity = _cosine(float(r["_score"]))
            # kNN always returns the k *nearest*, even when nothing is actually similar: drop weak matches.
            if similarity < self.settings.search_vector_min_similarity:
                continue
            hit = self._to_hit(r, parsed)
            hit.score = round(similarity, 4)
            hit.scores = {"vector": round(similarity, 4)}
            hits.append(hit)
        return hits

    def _fuse(self, keyword_hits: list[SearchHit], vector_hits: list[SearchHit]) -> list[SearchHit]:
        by_id = {h.source["segment_id"]: h for h in vector_hits} | {h.source["segment_id"]: h for h in keyword_hits}
        fused = reciprocal_rank_fusion(
            {
                "keyword": [h.source["segment_id"] for h in keyword_hits],
                "vector": [h.source["segment_id"] for h in vector_hits],
            },
            k=self.settings.search_rrf_k,
        )
        keyword_scores = {h.source["segment_id"]: h.score for h in keyword_hits}
        vector_scores = {h.source["segment_id"]: h.score for h in vector_hits}
        hits = []
        for segment_id, info in fused.items():
            hit = by_id[segment_id]
            # kNN always returns neighbours; with no keyword support, demand a stronger visual match.
            if "keyword" not in info["ranks"] and vector_scores[segment_id] < self.settings.search_vector_only_min_similarity:
                continue
            hit.scores = {"rrf": round(info["rrf"], 5)}
            for name, rank in info["ranks"].items():
                hit.scores[f"{name}_rank"] = rank
            if segment_id in keyword_scores:
                hit.scores["keyword"] = round(keyword_scores[segment_id], 4)
            if segment_id in vector_scores:
                hit.scores["vector"] = vector_scores[segment_id]
            hit.score = round(info["rrf"], 5)
            hits.append(hit)
        return hits

    # ---- shared ----------------------------------------------------------------------------------------
    def _to_hit(self, raw: dict[str, Any], parsed: ParsedQuery) -> SearchHit | None:
        doc = raw["_source"]
        highlight = next(iter(raw.get("highlight", {}).values()), [None])[0]
        start, end, match_score, matched_text = doc["start_sec"], doc["end_sec"], None, None
        if parsed.phrase and doc.get("words") is not None:
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
            scores={"keyword": round(float(raw["_score"] or 0.0), 4)},
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
