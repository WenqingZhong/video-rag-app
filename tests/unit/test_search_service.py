from unittest.mock import MagicMock

from src.config import Settings
from src.services.search import SearchService
from src.services.search import query_builder as qb

QUOTE_WORDS = [
    {"word": w, "start": s, "end": s + 0.3}
    for w, s in [
        ("answer", 8.0),
        ("is", 8.4),
        ("simple.", 8.8),
        ("AI", 9.48),
        ("is", 9.8),
        ("changing", 10.1),
        ("everything.", 10.5),
    ]
]


def doc(segment_id, video_id="v1", start=0.0, end=15.0, words=QUOTE_WORDS, kind="speech"):
    return {
        "segment_id": segment_id, "video_id": video_id, "kind": kind, "start_sec": start, "end_sec": end,
        "text": " ".join(w["word"] for w in words or []), "words": words, "video_s3_key": f"raw/{video_id}/source.mp4",
    }  # fmt: skip


def raw(source, score=5.0):
    return {"_source": source, "_score": score, "highlight": {"text": ["<em>AI</em> ..."]}}


def service(*responses):
    opensearch = MagicMock()
    opensearch.search.side_effect = [{"hits": {"hits": r}} for r in responses]
    return SearchService(opensearch, Settings(_env_file=None)), opensearch


def test_quote_uses_exact_phrase_first_and_returns_word_times():
    svc, os_ = service([raw(doc("s1"))])
    result = svc.search("the host says ‘AI is changing everything’", video_id="v1")

    assert result.strategy == "exact_phrase"
    hit = result.hits[0]
    assert (hit.match_start_sec, hit.match_end_sec, hit.match_score) == (9.48, 10.8, 1.0)
    body = os_.search.call_args.args[0]
    assert body["query"]["bool"]["must"][0] == {"match_phrase": {"text": {"query": "AI is changing everything", "slop": 0}}}
    assert {"term": {"video_id": "v1"}} in body["query"]["bool"]["filter"]
    assert {"term": {"kind": "speech"}} in body["query"]["bool"]["filter"]  # quotes search speech


def test_cascade_falls_back_to_slop_then_fuzzy():
    svc, os_ = service([], [], [raw(doc("s1"))])
    result = svc.search('"AI is changing everything"')
    assert result.strategy == "fuzzy_phrase" and os_.search.call_count == 3
    assert "fuzziness" in str(os_.search.call_args.args[0])


def test_overlapping_windows_collapse_to_one_moment():
    svc, _ = service(
        [raw(doc("s1", start=0, end=15), 5.0), raw(doc("s2", start=10, end=20), 4.0), raw(doc("s3", video_id="v2"), 3.0)]
    )
    result = svc.search('"AI is changing everything"')
    assert [h.source["segment_id"] for h in result.hits] == ["s1", "s3"]


def test_loose_opensearch_hit_without_real_phrase_is_dropped():
    unrelated = [{"word": "totally", "start": 1.0, "end": 1.2}, {"word": "different", "start": 1.3, "end": 1.6}]
    svc, _ = service([], [], [raw(doc("s1", words=unrelated))])
    result = svc.search('"AI is changing everything"')
    assert result.strategy == "none" and result.hits == []


def test_keyword_query_and_group_by_video():
    visual = [
        raw(doc(f"s{i}", video_id=vid, kind="visual", words=None, start=i * 7, end=i * 7 + 7), 9 - i)
        for i, vid in enumerate(["a", "a", "b"])
    ]
    svc, os_ = service(visual)
    result = svc.search("give me a clip of a dog", group_by_video=True)
    assert result.strategy == "keyword"
    assert [h.source["video_id"] for h in result.hits] == ["a", "b"]
    assert result.hits[0].match_start_sec == 0 and result.hits[0].match_score is None
    fields = os_.search.call_args.args[0]["query"]["bool"]["must"][0]["multi_match"]["fields"]
    assert fields == ["text.stemmed", "caption", "video_title^2"]  # never the stop-word-keeping exact field


LIBRARY = {"bool": {"must_not": {"exists": {"field": "owner_id"}}}}


def test_build_filters():
    assert qb.build_filters() == [LIBRARY]  # anonymous: the shared library only
    assert qb.build_filters(everyone=True) == []
    assert qb.build_filters(video_id="x", kind="visual", source="pexels") == [
        LIBRARY, {"term": {"video_id": "x"}}, {"term": {"kind": "visual"}}, {"term": {"video_source": "pexels"}},
    ]  # fmt: skip


def test_visibility_filter_adds_the_viewers_own_uploads():
    assert qb.visibility_filter(None) == LIBRARY
    assert qb.visibility_filter("web:abc") == {
        "bool": {"should": [LIBRARY, {"term": {"owner_id": "web:abc"}}], "minimum_should_match": 1}
    }


def test_search_service_filters_by_its_viewer():
    os_ = MagicMock()
    os_.search.return_value = {"hits": {"hits": []}}
    SearchService(os_, Settings(_env_file=None), viewer="tg:1").search("a dog", mode="keyword")
    filters = os_.search.call_args.args[0]["query"]["bool"]["filter"]
    assert filters[0] == qb.visibility_filter("tg:1")
