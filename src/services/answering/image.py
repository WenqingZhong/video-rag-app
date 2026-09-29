"""A photo in, the moment that looks most like it out: image search → clip → template answer."""

import time
from dataclasses import dataclass, field

from src.config import Settings
from src.services.answering.service import AnsweredClip
from src.services.answering.templates import explain_image
from src.services.clips import ClipService, clip_range
from src.services.search import SearchService


@dataclass
class ImageAnswer:
    status: str  # "answered" | "no_match"
    answer: str
    clips: list[AnsweredClip] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


class ImageAskService:
    def __init__(self, search: SearchService, clips: ClipService, settings: Settings):
        self.search = search
        self.clips = clips
        self.settings = settings

    def ask(self, image: bytes, video_id: str | None = None, source: str | None = None, max_clips: int = 1) -> ImageAnswer:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        result = self.search.search_image(
            image, video_id=video_id, source=source, size=max_clips, group_by_video=video_id is None
        )
        timings["search"] = round(time.perf_counter() - started, 3)
        if not result.hits:
            timings["total"] = timings["search"]
            return ImageAnswer("no_match", "Nothing in the library looks like this image.", timings=timings)

        t0 = time.perf_counter()
        answered = []
        for hit in result.hits[:max_clips]:
            doc = hit.source
            clip = clip_range(
                is_quote=False,
                match_start=hit.match_start_sec,
                match_end=hit.match_end_sec,
                segment_start=doc["start_sec"],
                segment_end=doc["end_sec"],
                frame_time=doc.get("frame_time_sec"),
                video_duration=doc.get("video_duration_sec"),
                padding=self.settings.clip_padding_sec,
                max_len=self.settings.clip_max_sec,
                visual_len=self.settings.clip_visual_sec,
            )
            key, cached = self.clips.get_or_cut(doc["video_id"], clip)
            answered.append(AnsweredClip(hit, clip, key, cached, explain_image(doc, clip.start_sec, clip.end_sec, hit.score)))
        timings["clips"] = round(time.perf_counter() - t0, 3)
        timings["total"] = round(time.perf_counter() - started, 3)
        return ImageAnswer("answered", answered[0].explanation, answered, timings)
