"""eval/intents.json and eval/no_answer.json back the request-understanding evaluation: keep them well-formed and in sync with saved results."""

import json
from pathlib import Path

EVAL = Path(__file__).resolve().parents[2] / "eval"
CATEGORIES = {"quote_marked", "quote_unmarked", "topic", "visual", "visual_exclude", "free_phrasing", "held_out", "no_subject"}


def test_intent_set_is_well_formed():
    requests = json.loads((EVAL / "intents.json").read_text())["requests"]
    assert len({r["request"].lower() for r in requests}) == len(requests), "duplicate requests"
    for r in requests:
        assert r["type"] in {"quote", "topic", "visual"} and r["category"] in CATEGORIES, r
        assert (r["text"] == "") == (r["category"] == "no_subject"), r  # '' means "no subject": only there
    assert sum(r["category"] == "held_out" for r in requests) >= 8


def test_no_answer_set_is_well_formed():
    queries = json.loads((EVAL / "no_answer.json").read_text())["queries"]
    assert len(queries) >= 10 and len(set(queries)) == len(queries)


def test_saved_results_match_the_sets():
    intents = json.loads((EVAL / "intents.json").read_text())["requests"]
    saved = json.loads((EVAL / "results" / "intents.json").read_text())
    assert saved["requests"] == len(intents), "re-run `make eval-intents` after editing eval/intents.json"
    search = json.loads((EVAL / "results" / "search.json").read_text())
    no_answer = json.loads((EVAL / "no_answer.json").read_text())["queries"]
    assert search["false_answers"]["understood"]["of"] == len(no_answer), "re-run `make eval` after editing eval/no_answer.json"
