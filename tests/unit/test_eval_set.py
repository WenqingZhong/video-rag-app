"""The labelled evaluation set is data that decisions rest on (docs/decisions/): keep it well-formed."""

import json
from collections import Counter
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[2] / "eval"
CATEGORIES = {"basic", "style", "detail", "synonym", "count", "color", "negation"}


@pytest.fixture(scope="module")
def queries() -> list[dict]:
    return json.loads((EVAL / "queries.json").read_text())["queries"]


def test_every_query_is_labelled(queries):
    for q in queries:
        assert q["query"].strip(), q
        assert q["relevant"], f"no relevant videos: {q['query']!r}"
        assert all(pid.isdigit() for pid in q["relevant"]), f"relevance must be Pexels ids: {q['query']!r}"
        assert len(set(q["relevant"])) == len(q["relevant"]), f"duplicate label: {q['query']!r}"
        assert q.get("category", "basic") in CATEGORIES, q


def test_queries_are_unique(queries):
    duplicates = [q for q, n in Counter(q["query"].lower() for q in queries).items() if n > 1]
    assert not duplicates


def test_hard_queries_cover_each_weakness(queries):
    """The caption-embedding decision (ADR 0001) depends on these categories being represented."""
    present = Counter(q.get("category", "basic") for q in queries)
    assert CATEGORIES <= set(present)
    assert present["basic"] >= 10 and sum(n for c, n in present.items() if c != "basic") >= 20


def test_saved_experiment_matches_the_eval_set(queries):
    """The saved evidence must come from this query set (re-run `make experiment-captions` after editing it)."""
    results = json.loads((EVAL / "results" / "caption_embeddings.json").read_text())
    assert results["queries"]["total"] == len(queries)
    assert {r["query"] for r in results["per_query"]} == {q["query"] for q in queries}
