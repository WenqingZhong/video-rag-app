"""Measure request understanding: the LLM (+ guards + rule safety net) vs the rule parser alone.

    make eval-intents      # = uv run python scripts/evaluate_intents.py --save eval/results/intents.json

Per request: type · text (phrase/topic/visual; "" = no subject) · exclusions · all three together.
Text is compared after lowercasing and dropping articles, punctuation and plural 's'.
Needs Ollama running with the understanding model.
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.services.understanding import QueryUnderstanding, make_query_understanding

_DROP = {"a", "an", "the"}


def norm(text: str | None) -> tuple[str, ...]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower().replace("’", "'"))
    return tuple(w.rstrip("s") if len(w) > 3 else w for w in words if w not in _DROP)


def score(system: QueryUnderstanding, requests: list[dict]) -> dict:
    rows, seconds = [], []
    for item in requests:
        result = system.understand(item["request"])
        seconds.append(result.seconds)
        intent = result.intent
        expected_text = item["text"]
        type_ok = intent.type == item["type"]
        text_ok = type_ok and (not intent.has_subject if expected_text == "" else norm(intent.text) == norm(expected_text))
        exclude_ok = sorted(map(norm, intent.exclude)) == sorted(map(norm, item.get("exclude", [])))
        rows.append(
            {
                "request": item["request"],
                "category": item["category"],
                "type": type_ok,
                "text": text_ok,
                "exclude": exclude_ok,
                "all": type_ok and text_ok and exclude_ok,
                "got": {**intent.model_dump(), "has_subject": intent.has_subject},
                "source": result.source,
                "notes": result.notes,
            }
        )
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    return {
        "accuracy": {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in ("type", "text", "exclude", "all")},
        "all_by_category": {c: round(sum(r["all"] for r in rs) / len(rs), 2) for c, rs in by_category.items()},
        "seconds": {"p50": round(statistics.median(seconds), 2), "max": round(max(seconds), 2)},
        "sources": {s: sum(1 for r in rows if r["source"] == s) for s in sorted({r["source"] for r in rows})},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="write the full report to this JSON file")
    args = parser.parse_args()

    requests = json.loads((ROOT / "eval" / "intents.json").read_text())["requests"]
    settings = Settings()
    systems = {"rules only": QueryUnderstanding(None), "LLM + guards + rules": make_query_understanding(settings)}
    systems["LLM + guards + rules"].understand("hi")  # load the model first, so latency isn't a cold start
    reports = {name: score(system, requests) for name, system in systems.items()}

    print(f"{len(requests)} labelled requests\n")
    print(f"{'system':<22}{'type':>7}{'text':>7}{'exclude':>9}{'all':>7}{'p50 s':>8}")
    for name, rep in reports.items():
        a = rep["accuracy"]
        print(f"{name:<22}{a['type']:>7.2f}{a['text']:>7.2f}{a['exclude']:>9.2f}{a['all']:>7.2f}{rep['seconds']['p50']:>8.2f}")
    categories = list(reports["rules only"]["all_by_category"])
    print("\nall-correct by category:")
    print(f"{'category':<16}{'n':>4}" + "".join(f"{name:>24}" for name in reports))
    for c in categories:
        n = sum(1 for r in requests if r["category"] == c)
        print(f"{c:<16}{n:>4}" + "".join(f"{rep['all_by_category'][c]:>24.2f}" for rep in reports.values()))
    llm = reports["LLM + guards + rules"]
    print(f"\nLLM system decided by: {llm['sources']}")
    print("\nrequests the LLM system got wrong:")
    for r in llm["rows"]:
        if not r["all"]:
            g = r["got"]
            text = g["phrase"] or g["topic"] or g["visual"]
            print(f"  [{r['category']}] {r['request'][:46]:<46} → {g['type']}: {text!r} exclude={g['exclude']} ({r['source']})")

    if args.save:
        report = {"model": settings.understanding_model, "requests": len(requests), "systems": reports}
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\nsaved → {args.save}")


if __name__ == "__main__":
    main()
