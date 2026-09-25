"""Locate a phrase inside a segment's words to get exact start/end seconds.

A search hit is a ~15 s window; users want the moment the phrase is said. Speech recognition makes
small mistakes ("changing" → "chaining", an extra "uh"), so matching is tolerant: each transcript token
is snapped to a phrase token it closely resembles, then the best-aligned span of words wins.
"""

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

TOKEN_SIMILARITY = 0.75  # "chaining" vs "changing" ≈ 0.75
MIN_SPAN_SCORE = 0.6


def normalize_tokens(text: str) -> list[str]:
    text = text.replace("’", "'").replace("‘", "'")
    # Strip accents (café → cafe) but turn other non-ASCII characters (em dashes, emoji) into spaces:
    # deleting them would glue neighbouring words together ("café—don't" → "cafedon't").
    decomposed = unicodedata.normalize("NFKD", text)
    text = "".join(c if c.isascii() else " " for c in decomposed if not unicodedata.combining(c)).lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text)


@dataclass(frozen=True)
class PhraseMatch:
    start_sec: float
    end_sec: float
    first_word: int
    last_word: int
    score: float  # character-level similarity of the quote to the matched words; 1.0 = exact

    def text(self, words: list[dict]) -> str:
        return " ".join(w["word"] for w in words[self.first_word : self.last_word + 1])


def _snap(token: str, phrase_tokens: set[str]) -> str:
    if token in phrase_tokens:
        return token
    best = max(phrase_tokens, key=lambda p: SequenceMatcher(None, token, p).ratio(), default=token)
    return best if SequenceMatcher(None, token, best).ratio() >= TOKEN_SIMILARITY else token


def locate_phrase(phrase: str, words: list[dict], slack: int = 2) -> PhraseMatch | None:
    target = normalize_tokens(phrase)
    if not target or not words:
        return None

    # Flatten words to tokens, remembering which word each token came from ("state-of-the-art" → 4 tokens, 1 word).
    tokens, raw_tokens, owner = [], [], []
    phrase_set = set(target)
    for i, word in enumerate(words):
        for token in normalize_tokens(word["word"]):
            tokens.append(_snap(token, phrase_set))
            raw_tokens.append(token)
            owner.append(i)
    if not tokens:
        return None

    best: tuple[float, int, int] | None = None  # (score, start_token, end_token)
    n = len(target)
    for start in range(len(tokens)):
        for length in range(max(1, n - slack), n + slack + 1):
            end = start + length
            if end > len(tokens):
                break
            score = SequenceMatcher(None, target, tokens[start:end], autojunk=False).ratio()
            if best is None or score > best[0] + 1e-9 or (abs(score - best[0]) < 1e-9 and length < best[2] - best[1]):
                best = (score, start, end)

    if best is None or best[0] < MIN_SPAN_SCORE:
        return None
    _, start, end = best
    first_word, last_word = owner[start], owner[end - 1]
    # Spans are *chosen* using snapped tokens (tolerant), but *reported* with an honest score against the
    # original spelling, so "chainging evrything" scores ~0.94, not a misleading 1.0.
    score = SequenceMatcher(None, " ".join(target), " ".join(raw_tokens[start:end]), autojunk=False).ratio()
    return PhraseMatch(
        start_sec=float(words[first_word]["start"]),
        end_sec=float(words[last_word]["end"]),
        first_word=first_word,
        last_word=last_word,
        score=round(score, 3),
    )
