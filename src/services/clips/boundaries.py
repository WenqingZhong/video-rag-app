"""Where a clip starts and ends, and its cache key. Pure functions: one set of rules for every caller."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClipRange:
    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 3)


def clip_range(
    *,
    is_quote: bool,
    match_start: float,
    match_end: float,
    segment_start: float,
    segment_end: float,
    frame_time: float | None,
    video_duration: float | None,
    padding: float,
    max_len: float,
) -> ClipRange:
    """Quote → the matched words ± padding (word-accurate). Visual/topic → the segment, at most max_len.

    A long visual segment is centred on its keyframe (the frame search actually matched).
    """
    limit = video_duration if video_duration and video_duration > 0 else max(segment_end, match_end)
    if is_quote:
        start, end = match_start - padding, match_end + padding
    else:
        start, end = segment_start, segment_end
        if end - start > max_len:
            centre = frame_time if frame_time is not None else (start + end) / 2
            start = max(segment_start, centre - max_len / 2)
            end = min(segment_end, start + max_len)
    start, end = max(0.0, start), min(limit, end)
    if end <= start:  # degenerate (e.g. a match at the very end): take a short window before it
        start = max(0.0, end - 1.0)
    # Rounded to 10 ms: the cache key is built from these, so identical requests reuse the same file.
    return ClipRange(round(start, 2), round(end, 2))


def clip_key(video_id: str, clip: ClipRange) -> str:
    return f"clips/{video_id}/{round(clip.start_sec * 1000):09d}-{round(clip.end_sec * 1000):09d}.mp4"
