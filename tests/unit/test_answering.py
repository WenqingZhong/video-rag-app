from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.services.answering import AskService, ask_for_subject, explain, mmss, no_match
from src.services.clips import ClipRange
from src.services.search import SearchHit, SearchResult
from src.services.search.query_parser import ParsedQuery
from src.services.understanding import Intent, Understanding

QUOTE = Intent(type="quote", phrase="AI is changing everything")


def test_templates():
    assert mmss(69.6) == "1:10"
    doc = {"kind": "speech", "video_title": "host_talk"}
    assert (
        explain(QUOTE, doc, 9.48, 10.86, "AI is changing everything.", 1.0)
        == 'At 0:09–0:11 in "host_talk": "AI is changing everything."'
    )
    assert explain(QUOTE, doc, 9.48, 10.86, "AI is chaining everything.", 0.96).startswith("Closest match at 0:09–0:11")
    pexels = {"kind": "visual", "video_title": "Perro", "video_source": "pexels", "video_author": "Elchino", "caption": "A dog."}
    assert (
        explain(Intent(type="visual", visual="a dog"), pexels, 0, 5.28, None, None)
        == '"Perro" (Pexels, by Elchino), 0:00–0:05: A dog.'
    )
    assert (
        no_match(Intent(type="visual", visual="a beach", exclude=["people"]))
        == 'No moment matched footage of "a beach" without people.'
    )
    assert (
        ask_for_subject(Intent(type="visual", exclude=["people"])) == "What would you like to see or hear? I'll leave out people."
    )


def speech_hit():
    doc = {"segment_id": "s1", "video_id": "v1", "kind": "speech", "start_sec": 0.0, "end_sec": 14.48, "video_title": "host_talk",
           "video_duration_sec": 17.44}  # fmt: skip
    return SearchHit(source=doc, score=1.0, match_start_sec=9.48, match_end_sec=10.86, match_score=1.0,
                     matched_text="AI is changing everything.", highlight=None)  # fmt: skip


@pytest.fixture
def parts():
    understanding, search, clips = MagicMock(), MagicMock(), MagicMock()
    clips.get_or_cut.return_value = ("clips/v1/000008730-000011610.mp4", False)
    return understanding, search, clips


def service(parts, intent):
    understanding, search, clips = parts
    understanding.understand.return_value = Understanding(intent, "llm", 1.2)
    return AskService(understanding, search, clips, Settings(_env_file=None))


def test_quote_is_cut_at_the_words_and_explained(parts):
    _, search, clips = parts
    search.search_intent.return_value = SearchResult(
        parsed=ParsedQuery("q", "p", "q"), strategy="exact_phrase", hits=[speech_hit()]
    )
    answer = service(parts, QUOTE).ask("the host says AI is changing everything", video_id="v1")

    assert answer.status == "answered"
    assert search.search_intent.call_args.kwargs["group_by_video"] is False  # one video: its best moments
    clips.get_or_cut.assert_called_once_with("v1", ClipRange(8.73, 11.61))  # the words ± 0.75 s
    assert answer.answer == 'At 0:09–0:11 in "host_talk": "AI is changing everything."'
    assert set(answer.timings) == {"understand", "search", "clips", "total"}


def test_library_search_is_one_moment_per_video(parts):
    _, search, _ = parts
    search.search_intent.return_value = SearchResult(parsed=ParsedQuery("q", None, "q"), strategy="none")
    service(parts, Intent(type="visual", visual="a dog")).ask("a dog")
    assert search.search_intent.call_args.kwargs["group_by_video"] is True


def test_no_match_cuts_nothing(parts):
    _, search, clips = parts
    search.search_intent.return_value = SearchResult(parsed=ParsedQuery("q", None, "q"), strategy="none")
    answer = service(parts, Intent(type="visual", visual="a dragon")).ask("a dragon")
    assert answer.status == "no_match" and answer.clips == [] and answer.answer.startswith("No moment matched")
    clips.get_or_cut.assert_not_called()


def test_no_subject_asks_instead_of_searching(parts):
    _, search, clips = parts
    answer = service(parts, Intent(type="visual", exclude=["people"])).ask("exclude people")
    assert answer.status == "needs_subject" and answer.answer.startswith("What would you like")
    search.search_intent.assert_not_called()
    clips.get_or_cut.assert_not_called()
