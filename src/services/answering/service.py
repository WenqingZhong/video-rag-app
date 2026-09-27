"""/ask: understand → search → pick the best moment(s) → cut clip(s) → template answer."""

import time
from dataclasses import dataclass, field

from src.config import Settings
from src.services.answering.templates import ask_for_subject, explain, no_match
from src.services.clips import ClipRange, ClipService, clip_range
from src.services.search import SearchHit, SearchService
from src.services.understanding import QueryUnderstanding, Understanding


@dataclass
class AnsweredClip:
    hit: SearchHit
    clip: ClipRange
    key: str
    cached: bool
    explanation: str


@dataclass
class Answer:
    understanding: Understanding
    status: str  # "answered" | "no_match" | "needs_subject"
    strategy: str
    answer: str
    clips: list[AnsweredClip] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


class AskService:
    def __init__(self, understanding: QueryUnderstanding, search: SearchService, clips: ClipService, settings: Settings):
        self.understanding = understanding
        self.search = search
        self.clips = clips
        self.settings = settings

    def ask(self, query: str, video_id: str | None = None, source: str | None = None, max_clips: int = 1) -> Answer:
        started = time.perf_counter()
        timings: dict[str, float] = {}

        def finish(**kwargs) -> Answer:
            timings["total"] = time.perf_counter() - started
            return Answer(timings={k: round(v, 3) for k, v in timings.items()}, **kwargs)

        understood = self.understanding.understand(query)
        intent = understood.intent
        timings["understand"] = understood.seconds

        if not intent.has_subject:  # "exclude people": nothing to search for → ask, don't guess
            return finish(understanding=understood, status="needs_subject", strategy="none", answer=ask_for_subject(intent))

        t0 = time.perf_counter()
        # Across the library: one moment per video. Inside one video: its best moments.
        result = self.search.search_intent(
            intent, video_id=video_id, source=source, size=max_clips, group_by_video=video_id is None
        )
        timings["search"] = time.perf_counter() - t0
        if not result.hits:
            return finish(understanding=understood, status="no_match", strategy=result.strategy, answer=no_match(intent))

        t0 = time.perf_counter()
        answered = []
        for hit in result.hits[:max_clips]:
            doc = hit.source
            is_quote = doc["kind"] == "speech" and hit.match_score is not None  # word-level times available
            clip = clip_range(
                is_quote=is_quote,
                match_start=hit.match_start_sec,
                match_end=hit.match_end_sec,
                segment_start=doc["start_sec"],
                segment_end=doc["end_sec"],
                frame_time=doc.get("frame_time_sec"),
                video_duration=doc.get("video_duration_sec"),
                padding=self.settings.clip_padding_sec,
                max_len=self.settings.clip_max_sec,
            )
            key, cached = self.clips.get_or_cut(doc["video_id"], clip)
            # Quotes describe the matched words; everything else describes the clip.
            start, end = (hit.match_start_sec, hit.match_end_sec) if is_quote else (clip.start_sec, clip.end_sec)
            answered.append(
                AnsweredClip(hit, clip, key, cached, explain(intent, doc, start, end, hit.matched_text, hit.match_score))
            )
        timings["clips"] = time.perf_counter() - t0
        return finish(
            understanding=understood, status="answered", strategy=result.strategy, answer=answered[0].explanation, clips=answered
        )
