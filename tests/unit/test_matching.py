from src.services.search import locate_phrase
from src.services.search.matching import normalize_tokens


def words(text: str, t0: float = 9.0, step: float = 0.4) -> list[dict]:
    return [
        {"word": w, "start": round(t0 + i * step, 2), "end": round(t0 + i * step + 0.35, 2)} for i, w in enumerate(text.split())
    ]


def test_normalize_handles_case_punctuation_accents_apostrophes():
    assert normalize_tokens("AI, is Changing café—don’t!") == ["ai", "is", "changing", "cafe", "don't"]


def test_exact_phrase_gives_word_level_times():
    w = words("My answer is simple. AI is changing everything. It changes")
    m = locate_phrase("AI is changing everything", w)
    assert (m.start_sec, m.end_sec, m.score) == (w[4]["start"], w[7]["end"], 1.0)
    assert m.text(w) == "AI is changing everything."


def test_misheard_word_still_located():
    w = words("simple. AI is chaining everything. It")
    m = locate_phrase("AI is changing everything", w)
    assert m is not None and m.text(w) == "AI is chaining everything."
    assert 0.9 < m.score < 1.0  # honest: close, not exact


def test_inserted_filler_word_lowers_score_but_matches():
    w = words("simple. AI is uh changing everything. It")
    m = locate_phrase("AI is changing everything", w)
    assert m.text(w) == "AI is uh changing everything." and 0.6 < m.score < 1.0


def test_absent_phrase_returns_none():
    assert locate_phrase("AI is changing everything", words("we talked about dogs and cats")) is None
    assert locate_phrase("anything", []) is None
