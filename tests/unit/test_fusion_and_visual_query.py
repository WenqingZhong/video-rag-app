import pytest

from src.services.search.fusion import reciprocal_rank_fusion
from src.services.search.visual_query import visual_query_text


def test_rrf_rewards_documents_ranked_by_both_retrievers():
    fused = reciprocal_rank_fusion({"keyword": ["a", "b", "c"], "vector": ["c", "d", "a"]}, k=60)
    assert list(fused)[:2] in (["a", "c"], ["c", "a"])  # in both lists → top two (tied: 1/61 + 1/63)
    assert fused["a"]["ranks"] == {"keyword": 1, "vector": 3}
    assert fused["b"]["rrf"] == pytest.approx(1 / 62)  # keyword only, rank 2
    assert fused["d"]["ranks"] == {"vector": 2}


def test_rrf_empty_list_is_ignored():
    assert list(reciprocal_rank_fusion({"keyword": ["x"], "vector": []})) == ["x"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("give me a clip of a dog", "a dog"),
        ("Show me footage of ocean waves", "ocean waves"),
        ("find videos of people cooking", "people cooking"),
        ("Could you show me a video where a cat sleeps?", "a cat sleeps"),
        ("a dog", "dog"),
        ("give me a clip", "clip"),
        ("dog", "dog"),
    ],
)
def test_visual_query_strips_request_words(query, expected):
    assert visual_query_text(query) == expected
