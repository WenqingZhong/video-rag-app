"""What the model is allowed to answer from: numbered, timed excerpts of transcripts and captions."""

from dataclasses import dataclass

from src.services.answering.templates import mmss
from src.services.clips.boundaries import join_words


@dataclass(frozen=True)
class Excerpt:
    video_id: str
    title: str
    kind: str  # "speech" (said) | "visual" (a keyframe caption: shown)
    start_sec: float
    end_sec: float
    text: str
    words: tuple = ()  # speech: Whisper words with times, to clip just the sentence that answers

    def render(self, number: int) -> str:
        said = "said" if self.kind == "speech" else "shown"
        return f"[{number}] {self.title} {mmss(self.start_sec)}–{mmss(self.end_sec)} ({said}): {self.text}"


def transcript_excerpts(video_id: str, title: str, windows: list[dict], chunk_sec: float = 20.0) -> list[Excerpt]:
    """A whole transcript, cut into ~20 s excerpts. Speech windows overlap (15 s every 10 s): rebuild the
    transcript from their word timings so no sentence appears twice."""
    words: dict[float, dict] = {}
    for window in windows:
        for word in window.get("words") or []:
            words.setdefault(round(word["start"], 2), word)
    excerpts: list[Excerpt] = []
    chunk: list[dict] = []
    for start in sorted(words):
        word = words[start]
        if chunk and word["start"] - chunk[0]["start"] >= chunk_sec and chunk[-1]["word"].rstrip().endswith((".", "?", "!")):
            excerpts.append(_excerpt(video_id, title, chunk))
            chunk = []
        chunk.append(word)
    if chunk:
        excerpts.append(_excerpt(video_id, title, chunk))
    return excerpts


def _excerpt(video_id: str, title: str, words: list[dict]) -> Excerpt:
    text = join_words([w["word"] for w in words])
    return Excerpt(video_id, title, "speech", words[0]["start"], words[-1]["end"], text, tuple(words))
