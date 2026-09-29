"""Where a clip starts and ends, and its cache key. Pure functions: one set of rules for every caller."""

import re
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
    visual_len: float | None = None,
    focus: tuple | None = None,
) -> ClipRange:
    """Where a clip starts and ends. A clip is a moment, not a scene:

    quote   the matched words ± padding (word-accurate)
    focus   a known moment inside the segment (e.g. the sentence that answers the question) ± padding
    visual  `visual_len` seconds centred on the keyframe (the frame search actually matched), inside its shot
    other   the segment, at most max_len
    """
    limit = video_duration if video_duration and video_duration > 0 else max(segment_end, match_end)
    if is_quote:
        start, end = match_start - padding, match_end + padding
    elif focus is not None:
        start, end = focus[0] - padding, min(focus[1] + padding, focus[0] + max_len)  # focus: (start, end, …)
    else:
        length = visual_len if visual_len and frame_time is not None else max_len
        start, end = segment_start, segment_end
        if end - start > length:
            centre = frame_time if frame_time is not None else (start + end) / 2
            start = max(segment_start, centre - length / 2)
            end = min(segment_end, start + length)
            start = max(segment_start, end - length)  # near the shot's end: keep the full length
    start, end = max(0.0, start), min(limit, end)
    if end <= start:  # degenerate (e.g. a match at the very end): take a short window before it
        start = max(0.0, end - 1.0)
    # Rounded to 10 ms: the cache key is built from these, so identical requests reuse the same file.
    return ClipRange(round(start, 2), round(end, 2))


_SMALL = {"a", "an", "the", "of", "on", "in", "at", "to", "for", "with", "and", "or", "is", "are", "it", "what", "how",
          "does", "do", "about", "that", "this", "she", "he", "they", "say", "says", "much", "many", "long"}  # fmt: skip


def _terms(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower())} - _SMALL


def join_words(tokens: list[str]) -> str:
    """Whisper splits "half-life" into "half" + "-life" and "it's" into "it" + "'s": glue those back on."""
    text = ""
    for token in (t.strip() for t in tokens):
        if not token:
            continue
        text += token if text and token[0] in "-'’.,?!%:;" else (" " + token if text else token)
    return text


def sentences(words: list[dict]) -> list[tuple[float, float, str]]:
    """Whisper words → sentences (start, end, text), split after . ? !"""
    out, current = [], []
    for word in words:
        current.append(word)
        if word["word"].rstrip().endswith((".", "?", "!")):
            out.append(current)
            current = []
    if current:
        out.append(current)
    return [(ws[0]["start"], ws[-1]["end"], join_words([w["word"] for w in ws])) for ws in out]


def best_sentence(words: list[dict] | None, about: str, min_len: float = 2.5) -> tuple[float, float, str] | None:
    """(start, end, text) of the sentence sharing the most words with `about` (a topic, a question and its answer).

    A very short sentence ("Here is my advice.") is joined with the next, so the clip says something.
    """
    if not words:
        return None
    wanted = _terms(about)
    spoken = sentences(words)
    scored = [(len(wanted & _terms(text)), i) for i, (_, _, text) in enumerate(spoken)]
    best, index = max(scored, key=lambda t: (t[0], -t[1]), default=(0, 0))
    if best == 0:
        return None
    start, end, text = spoken[index]
    if end - start < min_len and index + 1 < len(spoken):
        end, text = spoken[index + 1][1], f"{text} {spoken[index + 1][2]}"
    return start, end, text


def clip_key(video_id: str, clip: ClipRange) -> str:
    return f"clips/{video_id}/{round(clip.start_sec * 1000):09d}-{round(clip.end_sec * 1000):09d}.mp4"
