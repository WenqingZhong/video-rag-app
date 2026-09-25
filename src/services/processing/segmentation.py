"""Pure functions that turn raw signals (scene cuts, words) into search segments. No I/O: easy to unit test."""

from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class Shot:
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2


def build_shots(cut_times: list[float], duration: float, min_len: float, max_len: float) -> list[Shot]:
    """Turn scene-cut timestamps into shots.

    - cuts closer than `min_len` to the previous boundary are ignored (flicker, fast pans)
    - the tail shorter than `min_len` is merged into the previous shot
    - shots longer than `max_len` are split evenly, so each segment's keyframe represents it
    """
    boundaries = [0.0]
    for cut in sorted(cut_times):
        if min_len <= cut - boundaries[-1] and duration - cut >= min_len:
            boundaries.append(cut)
    boundaries.append(duration)

    shots: list[Shot] = []
    for start, end in pairwise(boundaries):
        length = end - start
        parts = max(1, int(-(-length // max_len)))  # ceil
        step = length / parts
        shots.extend(Shot(round(start + i * step, 3), round(start + (i + 1) * step, 3)) for i in range(parts))
    return shots


@dataclass(frozen=True)
class Word:
    word: str
    start: float
    end: float
    prob: float | None = None


def build_speech_windows(words: list[Word], window_sec: float, stride_sec: float) -> list[dict]:
    """Group words into overlapping windows.

    A new window starts every `stride_sec`, and each runs up to `window_sec`, so with 15/10 there are
    5 s of overlap: any phrase shorter than 5 s appears whole in at least one window.
    Word timestamps are kept so search can cut clips at exact word boundaries.
    """
    if not words:
        return []
    windows: list[dict] = []
    first = 0
    while first < len(words):
        window_start = words[first].start
        last = first
        while last + 1 < len(words) and words[last + 1].end - window_start <= window_sec:
            last += 1
        chunk = words[first : last + 1]
        windows.append(
            {
                "start_sec": round(chunk[0].start, 3),
                "end_sec": round(chunk[-1].end, 3),
                "text": "".join(w.word for w in chunk).strip(),
                "words": [{"word": w.word.strip(), "start": w.start, "end": w.end, "prob": w.prob} for w in chunk],
            }
        )
        if last == len(words) - 1:
            break
        next_first = first + 1
        while next_first <= last and words[next_first].start < window_start + stride_sec:
            next_first += 1
        first = next_first
    return windows
