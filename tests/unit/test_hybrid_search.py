from unittest.mock import MagicMock

import httpx
import pytest

from src.config import Settings
from src.services.search import SearchService
from src.services.search.service import SearchUnavailable


def doc(segment_id, video_id, kind="visual", start=0.0, end=7.0, caption=None):
    return {"segment_id": segment_id, "video_id": video_id, "kind": kind, "start_sec": start, "end_sec": end, "caption": caption}


def raw(source, score):
    return {"_source": source, "_score": score}


def cosine_score(cosine):
    return (1 + cosine) / 2  # how OpenSearch reports cosinesimil kNN scores


def make(keyword_hits, vector_hits, embedder=True):
    opensearch = MagicMock()

    def search(body):
        return {"hits": {"hits": vector_hits if "knn" in body["query"] else keyword_hits}}

    opensearch.search.side_effect = search
    emb = None
    if embedder:
        emb = MagicMock()
        emb.embed_texts.return_value = [[0.1, 0.2]]
    return SearchService(opensearch, Settings(_env_file=None), embedder=emb), opensearch, emb


def test_hybrid_fuses_keyword_and_vector_with_rrf():
    keyword = [raw(doc("title_match", "v1"), 3.0)]
    vector = [raw(doc("puppies", "v2"), cosine_score(0.30)), raw(doc("title_match", "v1"), cosine_score(0.25))]
    svc, _, emb = make(keyword, vector)

    result = svc.search("give me a clip of a dog")

    assert result.strategy == "hybrid" and result.visual_query == "a dog"
    emb.embed_texts.assert_called_once_with(["a dog"])  # request words stripped before CLIP
    top = result.hits[0]
    assert top.source["segment_id"] == "title_match"  # ranked by BOTH retrievers → first
    assert top.scores["keyword_rank"] == 1 and top.scores["vector_rank"] == 2 and top.scores["vector"] == 0.25
    assert [h.source["segment_id"] for h in result.hits] == ["title_match", "puppies"]


def test_weak_vector_matches_are_dropped():
    vector = [raw(doc("ocean", "v3"), cosine_score(0.05))]  # nearest, but not similar
    svc, _, _ = make([], vector)
    assert svc.search("a dog", mode="vector").strategy == "none"


def test_auto_falls_back_to_keyword_when_embedder_is_down():
    svc, _, emb = make([raw(doc("s1", "v1"), 2.0)], [])
    emb.embed_texts.side_effect = httpx.ConnectError("embedder down")
    result = svc.search("a dog")
    assert result.strategy == "keyword" and len(result.hits) == 1


def test_forced_vector_mode_reports_unavailable():
    svc, _, _ = make([], [], embedder=False)
    with pytest.raises(SearchUnavailable):
        svc.search("a dog", mode="vector")


def test_vector_query_carries_filters_and_excludes_vectors_from_results():
    svc, opensearch, _ = make([], [raw(doc("s1", "v1"), cosine_score(0.3))])
    svc.search("a dog", mode="vector", source="pexels")
    body = opensearch.search.call_args.args[0]
    assert body["query"]["knn"]["image_embedding"]["filter"] == {"bool": {"filter": [{"term": {"video_source": "pexels"}}]}}
    assert body["_source"] == {"excludes": ["image_embedding"]}


def test_speech_only_search_skips_vectors_and_quotes_keep_phrase_path():
    svc, _, emb = make([raw(doc("s1", "v1", kind="speech"), 1.0)], [])
    assert svc.search("future of work", kind="speech").strategy == "keyword"
    emb.embed_texts.assert_not_called()
