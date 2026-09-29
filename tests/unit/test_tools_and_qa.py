import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from src.config import Settings
from src.models import Video, VideoSource, VideoStatus
from src.repositories import VideoRepository
from src.services.agent import Toolbox
from src.services.answering import ImageAskService
from src.services.clips.boundaries import join_words
from src.services.llm import OllamaChat
from src.services.qa import NOT_FOUND, Excerpt, LLMAnswerer, QAService, numbers_in, retrieval_text, transcript_excerpts
from src.services.search import SearchHit, SearchResult, SearchService
from src.services.search.query_parser import ParsedQuery


# ---- image search ----------------------------------------------------------------------------------------------
def knn_hit(video_id: str, cosine: float) -> dict:
    doc = {"segment_id": f"s-{video_id}", "video_id": video_id, "kind": "visual", "start_sec": 0.0, "end_sec": 4.0,
           "frame_time_sec": 2.0, "video_title": video_id}  # fmt: skip
    return {"_score": (1 + cosine) / 2, "_source": doc}  # OpenSearch reports cosinesimil as (1 + cos) / 2


def test_image_search_keeps_only_close_keyframes():
    opensearch, embedder = MagicMock(), MagicMock()
    embedder.embed_images.return_value = [[1.0, 0.0]]
    opensearch.search.return_value = {"hits": {"hits": [knn_hit("dog", 0.72), knn_hit("cat", 0.40)]}}
    service = SearchService(opensearch, Settings(_env_file=None), embedder=embedder)

    result = service.search_image(b"jpeg", size=3)

    assert [h.source["video_id"] for h in result.hits] == ["dog"]  # 0.40 is below the image cut-off
    assert result.hits[0].scores == {"image": 0.72} and result.strategy == "image"
    body = opensearch.search.call_args.args[0]
    assert body["query"]["knn"]["image_embedding"]["filter"]["bool"]["filter"] == [{"term": {"kind": "visual"}}]


def test_image_with_nothing_similar_returns_no_clip():
    search, clips = MagicMock(), MagicMock()
    search.search_image.return_value = SearchResult(parsed=ParsedQuery("<image>", None, ""), strategy="none")
    answer = ImageAskService(search, clips, Settings(_env_file=None)).ask(b"jpeg")
    assert answer.status == "no_match" and answer.clips == [] and not clips.get_or_cut.called


# ---- excerpts ----------------------------------------------------------------------------------------------------
def test_whisper_pieces_are_glued_back():
    assert (
        join_words(["a", "half", "-life", "of", "5", "hours.", "It", "'s", "wake", "-up"])
        == "a half-life of 5 hours. It's wake-up"
    )


def test_overlapping_windows_become_one_transcript_in_sentence_chunks():
    words = [{"word": w, "start": float(i), "end": i + 0.5} for i, w in enumerate(["One.", "Two.", "Three.", "Four.", "Five."])]
    windows = [{"words": words[:4]}, {"words": words[2:]}]  # windows overlap: "Three." and "Four." twice
    excerpts = transcript_excerpts("v1", "talk", windows, chunk_sec=2.0)
    assert [e.text for e in excerpts] == ["One. Two.", "Three. Four.", "Five."]
    assert (excerpts[1].start_sec, excerpts[1].end_sec) == (2.0, 3.5)


def test_question_words_are_dropped_for_retrieval():
    assert retrieval_text("What does the host say about caffeine?") == "caffeine"


def test_numbers_are_read_as_digits():
    assert numbers_in("about ninety minutes, 4 to 6 cycles, 2.5 hours") == {"90", "4", "6", "2.5"}


# ---- question answering ------------------------------------------------------------------------------------------
TALK = [Excerpt("v1", "sleep_talk", "speech", 10.0, 20.0, "Each cycle lasts about 90 minutes."),
        Excerpt("v1", "sleep_talk", "speech", 50.0, 60.0, "Caffeine has a half-life of about 5 hours.")]  # fmt: skip


def qa_with(reply: dict | str, excerpts=TALK, clips=None) -> QAService:
    llm = LLMAnswerer(OllamaChat("http://ollama", "qwen2.5vl:3b"))

    def handler(request):
        content = json.dumps(reply) if isinstance(reply, dict) else reply
        return httpx.Response(200, json={"message": {"content": content}, "prompt_eval_count": 300, "eval_count": 30})

    llm.chat.http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    service = QAService(MagicMock(), MagicMock(), llm, Settings(_env_file=None), clips=clips)
    service.excerpts = lambda question, video_id=None: list(excerpts)
    return service


def test_a_cited_answer_is_returned_with_its_moment_as_a_clip():
    spoken = "Let's start with cycles. Each cycle lasts about 90 minutes. Most adults go through several."
    words = tuple({"word": w, "start": 10.0 + i, "end": 10.6 + i} for i, w in enumerate(spoken.split()))
    excerpt = Excerpt("v1", "sleep_talk", "speech", 10.0, 24.6, spoken, words)
    clips = MagicMock()
    clips.get_or_cut.return_value = ("clips/v1/x.mp4", False)
    service = qa_with({"found": True, "answer": "About ninety minutes.", "cited": [1]}, excerpts=[excerpt], clips=clips)
    result = service.answer("How long is a sleep cycle?")
    assert result.status == "answered" and result.citations == [excerpt] and result.llm_outcome == "accepted"
    clip = clips.get_or_cut.call_args.args[1]
    # "Each cycle lasts about 90 minutes." is words 4–9 (14.0–19.6 s): just that sentence, ± 0.75 s
    assert (clip.start_sec, clip.end_sec) == (13.25, 20.35)


@pytest.mark.parametrize(
    ("reply", "outcome", "note"),
    [
        ({"found": False, "answer": "", "cited": []}, "not_found", "no answer"),
        ({"found": True, "answer": "About 90 minutes.", "cited": [7]}, "rejected", "no valid citation"),
        ({"found": True, "answer": "About 8 hours.", "cited": [2]}, "rejected", "numbers not in the cited excerpts"),
        ("not json", "invalid", "reply invalid"),
    ],
)
def test_guards_turn_unsupported_answers_into_not_found(reply, outcome, note):
    result = qa_with(reply).answer("question")
    assert result.status == "not_found" and result.answer == NOT_FOUND and result.llm_outcome == outcome
    assert note in result.notes[0]


def test_no_excerpts_means_no_model_call():
    service = qa_with({"found": True, "answer": "x", "cited": [1]}, excerpts=[])
    result = service.answer("What is the capital of France?")
    assert result.status == "not_found" and result.llm_call is None


def test_one_video_is_answered_from_its_whole_transcript_and_captions(database):
    with database.get_session() as session:
        repo = VideoRepository(session)
        video = repo.create(source=VideoSource.UPLOAD, title="talk", status=VideoStatus.READY)
        words = [{"word": w, "start": float(i), "end": i + 0.5} for i, w in enumerate(["Sleep", "in", "cycles."])]
        repo.replace_segments(video.id, [
            {"kind": "speech", "idx": 0, "start_sec": 0.0, "end_sec": 3.0, "text": "Sleep in cycles.", "words": words},
            {"kind": "visual", "idx": 0, "start_sec": 0.0, "end_sec": 3.0, "caption": "A grey background."},
        ])  # fmt: skip
        session.commit()
        video_id = video.id
    service = QAService(database, MagicMock(), None, Settings(_env_file=None))
    excerpts = service.excerpts("anything", video_id=video_id)
    assert [(e.kind, e.text) for e in excerpts] == [("speech", "Sleep in cycles."), ("visual", "A grey background.")]


# ---- tools -------------------------------------------------------------------------------------------------------
@pytest.fixture
def toolbox(database):
    storage = MagicMock()
    storage.presigned_url.side_effect = lambda key: f"http://s3/{key}"
    return Toolbox(database, storage, MagicMock(), MagicMock(), MagicMock(), make_ingestion=MagicMock(), pexels=None)


def test_unknown_tools_and_bad_arguments_are_results_not_crashes(toolbox):
    assert not toolbox.run("delete_everything", {}).ok
    assert "Wrong arguments" in toolbox.run("find_clip", {"query": "a dog"}).summary  # the argument is called "request"


def test_a_failing_tool_reports_the_failure(toolbox):
    toolbox.ask.ask.side_effect = RuntimeError("opensearch down")
    result = toolbox.run("find_clip", {"request": "a dog"})
    assert not result.ok and "failed" in result.summary


def test_no_match_suggests_pexels(toolbox):
    understanding = SimpleNamespace(intent=SimpleNamespace(text="a horse", model_dump=lambda: {"visual": "a horse"}))
    toolbox.ask.ask.return_value = SimpleNamespace(
        status="no_match", answer="No moment matched…", clips=[], understanding=understanding
    )
    result = toolbox.run("find_clip", {"request": "a horse"})
    assert result.ok and "Pexels" in result.summary and result.data["status"] == "no_match"


def test_fetching_needs_a_pexels_key(toolbox):
    assert not toolbox.run("fetch_from_pexels", {"query": "horse"}).ok


def test_check_videos_counts_ready_and_pending(toolbox, database):
    with database.get_session() as session:
        repo = VideoRepository(session)
        ready = repo.create(source=VideoSource.PEXELS, source_id="1", status=VideoStatus.READY)
        busy = repo.create(source=VideoSource.PEXELS, source_id="2", status=VideoStatus.PROCESSING)
        session.commit()
        ids = [ready.id, busy.id, "gone"]
    result = toolbox.run("check_videos", {"video_ids": ids})
    assert (result.data["ready"], result.data["pending"], result.data["failed"]) == (1, 1, 1)


def test_answer_question_summary_names_its_sources(toolbox):
    toolbox.qa.answer.return_value = SimpleNamespace(
        status="answered", answer="About 90 minutes.", citations=[TALK[0]], clip_key=None, clip=None
    )
    result = toolbox.run("answer_question", {"question": "How long is a sleep cycle?"})
    assert result.summary == 'About 90 minutes. (from "sleep_talk" 0:10–0:20)'


def test_list_videos(toolbox, database):
    with database.get_session() as session:
        session.add(Video(source="upload", title="sleep_talk", status="ready", duration_sec=102.0, language="en", tags=[]))
        session.commit()
    result = toolbox.run("list_videos", {})
    assert result.data["total"] == 1 and "sleep_talk (102 s, speech)" in result.summary


def test_image_search_hit_is_a_plain_search_hit():
    # the answering code treats image hits like any other visual hit
    assert SearchHit.__dataclass_fields__.keys() >= {"source", "score", "match_start_sec", "match_end_sec"}
