"""Question answering over what videos say and show: retrieve excerpts → the model answers from them only
→ guards check the answer against the excerpts → the cited moment as a clip.

The first place the app *generates* text. The guards follow request understanding's idea: the model proposes, plain code checks.
"""

import logging
import re
import time
from dataclasses import dataclass, field

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.repositories import VideoRepository
from src.services.clips import ClipRange, ClipService, best_sentence, clip_range
from src.services.llm import ModelError
from src.services.qa.excerpts import Excerpt, transcript_excerpts
from src.services.qa.llm import LLMAnswerer
from src.services.search import SearchService
from src.services.tracing import span
from src.services.understanding.llm import InvalidReply
from src.services.usage import LLMCall, Outcome, UsageRecorder

logger = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find that in the videos."
# Words that make a question a question but don't say what it's about: dropped before keyword retrieval.
_QUESTION_WORDS = {
    "what", "whats", "which", "who", "whom", "when", "where", "why", "how", "does", "do", "did", "is", "are", "was", "were",
    "the", "a", "an", "about", "say", "says", "said", "tell", "tells", "talk", "talks", "mention", "mentions", "host",
    "speaker", "video", "videos", "clip", "in", "of", "to", "me", "they", "he", "she", "it", "there", "any", "can", "you",
}  # fmt: skip
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
}  # fmt: skip


@dataclass
class QAAnswer:
    status: str  # "answered" | "not_found" | "unavailable"
    answer: str
    citations: list[Excerpt] = field(default_factory=list)
    clip_key: str | None = None
    clip: ClipRange | None = None
    notes: list[str] = field(default_factory=list)  # what the guards found
    llm_call: LLMCall | None = None
    llm_outcome: Outcome | None = None
    cost_usd: float = 0.0
    excerpts: int = 0  # how many excerpts the model was shown
    timings: dict[str, float] = field(default_factory=dict)


def retrieval_text(question: str) -> str:
    words = re.findall(r"[a-z0-9']+", question.lower())
    kept = [w for w in words if w.strip("'") not in _QUESTION_WORDS]
    return " ".join(kept) or question


def numbers_in(text: str) -> set[str]:
    """Digits, with number words read as digits ("ninety minutes" and "90 minutes" both give {"90"})."""
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())
    return {_NUMBER_WORDS.get(t, t) for t in tokens if t.isdigit() or t in _NUMBER_WORDS or re.fullmatch(r"\d+\.\d+", t)}


class QAService:
    def __init__(
        self,
        database: BaseDatabase,
        search: SearchService,
        llm: LLMAnswerer | None,
        settings: Settings,
        recorder: UsageRecorder | None = None,
        clips: ClipService | None = None,
    ):
        self.database = database
        self.search = search
        self.llm = llm
        self.settings = settings
        self.recorder = recorder
        self.clips = clips  # None: no clip (e.g. tests, evaluation)

    # ---- retrieval -----------------------------------------------------------------------------------
    def excerpts(self, question: str, video_id: str | None = None) -> list[Excerpt]:
        """One video: its whole transcript (if short enough) plus its keyframe captions. The library: the best matches."""
        limit = self.settings.qa_max_excerpts
        if video_id is not None:
            with self.database.get_session() as session:
                repo = VideoRepository(session)
                video = repo.get_visible(video_id, getattr(self.search, "viewer", None))  # same visibility as search
                if video is None:
                    return []
                title = video.title or video.original_filename or video_id
                windows = [{"words": s.words} for s in repo.list_segments(video_id, kind="speech")]
                captions = [
                    Excerpt(video_id, title, "visual", s.start_sec, s.end_sec, s.caption)
                    for s in repo.list_segments(video_id, kind="visual")
                    if s.caption
                ]
            spoken = transcript_excerpts(video_id, title, windows)
            if len(spoken) + len(captions) <= limit:
                return spoken + captions
            return self._rank(question, spoken + captions, limit)

        result = self.search.search(retrieval_text(question), size=limit, mode="keyword")
        found = []
        for hit in result.hits:
            doc = hit.source
            text = doc.get("text") if doc["kind"] == "speech" else doc.get("caption")
            if text:
                title = doc.get("video_title") or doc["video_id"]
                words = tuple(doc.get("words") or ()) if doc["kind"] == "speech" else ()
                found.append(Excerpt(doc["video_id"], title, doc["kind"], doc["start_sec"], doc["end_sec"], text, words))
        return found

    @staticmethod
    def _rank(question: str, excerpts: list[Excerpt], limit: int) -> list[Excerpt]:
        """A long video: keep the `limit` excerpts sharing the most question words, shown in time order."""
        wanted = set(retrieval_text(question).split())
        best = sorted(excerpts, key=lambda e: -len(wanted & set(re.findall(r"[a-z0-9']+", e.text.lower()))))[:limit]
        return sorted(best, key=lambda e: (e.video_id, e.start_sec))

    # ---- answering -----------------------------------------------------------------------------------
    def answer(self, question: str, video_id: str | None = None) -> QAAnswer:
        with span("qa", video_id=video_id) as step:
            result = self._answer(question, video_id)
            step.set(status=result.status, outcome=result.llm_outcome, excerpts=result.excerpts, cited=len(result.citations))
            return result

    def _answer(self, question: str, video_id: str | None) -> QAAnswer:
        started = time.perf_counter()
        timings: dict[str, float] = {}

        def done(result: QAAnswer) -> QAAnswer:
            timings["total"] = round(time.perf_counter() - started, 3)
            result.timings = timings
            if result.llm_call is not None and self.recorder is not None:
                result.cost_usd = self.recorder.record(result.llm_call, result.llm_outcome or "accepted")
            return result

        with span("qa.retrieve"):
            excerpts = self.excerpts(question, video_id)
        timings["retrieve"] = round(time.perf_counter() - started, 3)
        if not excerpts:
            return done(QAAnswer("not_found", NOT_FOUND, notes=["no excerpts matched the question"]))
        if self.llm is None:
            return done(
                QAAnswer("unavailable", "Question answering needs the language model, which is disabled.", excerpts=len(excerpts))
            )

        rendered = "\n".join(e.render(i) for i, e in enumerate(excerpts, start=1))
        t0 = time.perf_counter()
        with span("llm.answer", model=self.llm.model, excerpts=len(excerpts)) as llm_span:
            try:
                reply, call = self.llm.answer(question, rendered)
            except InvalidReply as exc:
                llm_span.fail(exc)
                timings["llm"] = round(time.perf_counter() - t0, 3)
                return done(QAAnswer("not_found", NOT_FOUND, notes=[f"reply invalid: {exc}"], llm_call=exc.call,
                                     llm_outcome="invalid", excerpts=len(excerpts)))  # fmt: skip
            except ModelError as exc:
                llm_span.fail(exc)
                logger.warning("question answering failed: %s", exc)
                return done(QAAnswer("unavailable", "The language model is unavailable right now.", notes=[str(exc)],
                                     excerpts=len(excerpts)))  # fmt: skip
            llm_span.set(prompt_tokens=call.prompt_tokens, output_tokens=call.output_tokens)
        timings["llm"] = round(time.perf_counter() - t0, 3)

        result = self._check(reply, excerpts)
        result.llm_call, result.excerpts = call, len(excerpts)
        if result.status == "answered" and self.clips is not None:
            t0 = time.perf_counter()
            first = result.citations[0]
            # The sentence that answers, not the whole ~20 s excerpt (a visual citation: its keyframe's moment).
            focus = best_sentence(list(first.words), f"{question} {result.answer}")
            clip = clip_range(
                is_quote=False,
                match_start=first.start_sec,
                match_end=first.end_sec,
                segment_start=first.start_sec,
                segment_end=first.end_sec,
                frame_time=(first.start_sec + first.end_sec) / 2,
                video_duration=None,
                padding=self.settings.clip_padding_sec,
                max_len=self.settings.clip_max_sec,
                visual_len=self.settings.clip_visual_sec,
                focus=focus,
            )
            result.clip_key, _ = self.clips.get_or_cut(first.video_id, clip)
            result.clip = clip
            timings["clip"] = round(time.perf_counter() - t0, 3)
        return done(result)

    @staticmethod
    def _check(reply, excerpts: list[Excerpt]) -> QAAnswer:
        """Guards: the model must cite excerpts it was given, and every number it states must be in them."""
        if not reply.found or not reply.answer.strip():
            return QAAnswer("not_found", NOT_FOUND, notes=["the model found no answer in the excerpts"], llm_outcome="not_found")
        cited = [excerpts[n - 1] for n in dict.fromkeys(reply.cited) if 1 <= n <= len(excerpts)]
        if not cited:
            return QAAnswer("not_found", NOT_FOUND, notes=[f"no valid citation (cited {reply.cited})"], llm_outcome="rejected")
        source_numbers = numbers_in(" ".join(e.text for e in cited))
        invented = numbers_in(reply.answer) - source_numbers
        if invented:
            return QAAnswer(
                "not_found", NOT_FOUND, notes=[f"numbers not in the cited excerpts: {sorted(invented)}"], llm_outcome="rejected"
            )
        return QAAnswer("answered", reply.answer.strip(), citations=cited, llm_outcome="accepted")
