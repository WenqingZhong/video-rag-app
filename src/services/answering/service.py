"""/ask: understand → search → pick the best moment(s) → cut clip(s) → template answer."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.config import Settings
from src.services.answering.templates import ask_for_subject, explain, no_match
from src.services.clips import ClipRange, ClipService, best_sentence, clip_range
from src.services.search import SearchHit, SearchService
from src.services.tracing import span
from src.services.understanding import QueryUnderstanding, Understanding

if TYPE_CHECKING:
    from src.services.answering.cache import AnswerCache


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
    cache: dict[str, str] = field(default_factory=dict)  # {"answer": hit|miss|off, "understanding": hit|miss|off|skipped}


class AskService:
    def __init__(
        self,
        understanding: QueryUnderstanding,
        search: SearchService,
        clips: ClipService,
        settings: Settings,
        answers: "AnswerCache | None" = None,
    ):
        self.understanding = understanding
        self.search = search
        self.clips = clips
        self.settings = settings
        self.answers = answers  # None: every request is answered from scratch

    def ask(self, query: str, video_id: str | None = None, source: str | None = None, max_clips: int = 1) -> Answer:
        started = time.perf_counter()
        key = self.answers.key(query, video_id, source, max_clips) if self.answers else None
        if key is not None:
            with span("cache.answer") as step:
                cached = self._cached(key)
                step.set(hit=cached is not None)
            if cached is not None:
                seconds = round(time.perf_counter() - started, 3)
                cached.timings = {"cache": seconds, "total": seconds}
                return cached

        answer = self._answer(query, video_id, source, max_clips)
        understood = answer.understanding
        # An answer made with the rules only because the model failed is not reused: next time the model may work.
        model_failed = self.understanding.llm is not None and understood.llm_call is None and not understood.cached
        if key is not None and not model_failed:
            self.answers.put(key, answer)
        answer.cache = {
            "answer": "off" if key is None else "miss",
            "understanding": _understanding_cache_state(self.understanding, understood),
        }
        return answer

    def _cached(self, key: str) -> Answer | None:
        data = self.answers.get(key)
        if data is None:
            return None
        # Clip files are cached in S3, not in Redis: if one was deleted since, answer afresh.
        if not all(self.clips.storage.exists(c["key"]) for c in data["clips"]):
            return None
        fields = data["understanding"]
        avoided = fields.pop("avoided")
        understood = Understanding(seconds=0.0, cached=True, avoided_call=avoided, **fields)
        recorder = self.understanding.recorder
        if avoided is not None and recorder is not None:
            understood.saved_cost_usd = recorder.record_saved(avoided)
        clips = [AnsweredClip(c["hit"], c["clip"], c["key"], True, c["explanation"]) for c in data["clips"]]
        return Answer(
            understood,
            data["status"],
            data["strategy"],
            data["answer"],
            clips,
            cache={"answer": "hit", "understanding": "skipped"},
        )

    def _answer(self, query: str, video_id: str | None, source: str | None, max_clips: int) -> Answer:
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
            # A topic found in speech: the sentence about it, not the whole 15 s window.
            focus = best_sentence(doc.get("words"), intent.text) if doc["kind"] == "speech" and not is_quote else None
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
                visual_len=self.settings.clip_visual_sec,
                focus=focus,
            )
            key, cached = self.clips.get_or_cut(doc["video_id"], clip)
            # Quotes describe the matched words; everything else describes the clip.
            start, end = (hit.match_start_sec, hit.match_end_sec) if is_quote else (clip.start_sec, clip.end_sec)
            # A topic clip quotes its sentence, not the whole transcript window.
            said = focus[2] if focus else hit.matched_text
            answered.append(AnsweredClip(hit, clip, key, cached, explain(intent, doc, start, end, said, hit.match_score)))
        timings["clips"] = time.perf_counter() - t0
        return finish(
            understanding=understood, status="answered", strategy=result.strategy, answer=answered[0].explanation, clips=answered
        )


def _understanding_cache_state(understanding: QueryUnderstanding, understood: Understanding) -> str:
    if understanding.cache is None or understanding.llm is None:
        return "off"
    return "hit" if understood.cached else "miss"
