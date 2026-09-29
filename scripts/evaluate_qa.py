"""Question answering: right facts, right citations, and "not found" when the videos don't say.

    make eval-qa      # = uv run python scripts/evaluate_qa.py --save eval/results/qa.json

Calls POST /api/v1/answer on the running API for each labelled question in eval/questions.json:
  status        answered vs not_found as expected
  facts         the answer contains every required fact (for questions that should be answered)
  citation      one cited excerpt contains the labelled text (the answer points at the right moment)
  made up       answered a question the videos don't answer (the failure that matters most)
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text

from src.config import Settings
from src.services.qa import numbers_in

API = "http://localhost:8000/api/v1"


def contains(answer: str, alternatives: list[str]) -> bool:
    low = answer.lower()
    for alt in alternatives:
        if alt.isdigit() and alt in numbers_in(answer):  # "ninety" counts for "90"
            return True
        if re.search(r"\b" + re.escape(alt.lower()), low):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    settings = Settings()
    engine = create_engine(settings.postgres_database_url)
    with engine.connect() as c:
        by_title = {
            r.title: r.id
            for r in c.execute(
                text("SELECT DISTINCT ON (title) title, id FROM videos WHERE status = 'ready' ORDER BY title, created_at DESC")
            )
        }
    questions = json.loads((ROOT / "eval" / "questions.json").read_text())["questions"]

    http = httpx.Client(timeout=180)
    http.post(f"{API}/answer", json={"question": "warm up"})  # load the model first
    rows = []
    for q in questions:
        video_id = by_title[q["video"]] if q["video"] else None
        started = time.perf_counter()
        body = http.post(f"{API}/answer", json={"question": q["question"], "video_id": video_id}).raise_for_status().json()
        seconds = time.perf_counter() - started
        answered = body["status"] == "answered"
        facts = answered and all(contains(body["answer"], group) for group in q.get("must", []))
        cited = answered and any(q.get("cite", "").lower() in c["text"].lower() for c in body["citations"])
        rows.append(
            {
                **q,
                "status": body["status"],
                "answer": body["answer"],
                "notes": body["notes"],
                "status_ok": body["status"] == q["expect"],
                "facts_ok": facts,
                "citation_ok": cited,
                "seconds": round(seconds, 2),
                "tokens": body["usage"]["total_tokens"],
                "outcome": body["usage"]["outcome"],
            }
        )

    should_answer = [r for r in rows if r["expect"] == "answered"]
    should_not = [r for r in rows if r["expect"] == "not_found"]
    summary = {
        "questions": len(rows),
        "status_accuracy": round(sum(r["status_ok"] for r in rows) / len(rows), 3),
        "answered_correctly": round(sum(r["facts_ok"] for r in should_answer) / len(should_answer), 3),
        "cited_correctly": round(sum(r["citation_ok"] for r in should_answer) / len(should_answer), 3),
        "made_up": sum(r["status"] == "answered" for r in should_not),
        "should_not_answer": len(should_not),
        "refused_answerable": sum(r["status"] != "answered" for r in should_answer),
        "p50_seconds": round(statistics.median(r["seconds"] for r in rows), 2),
        "tokens_per_question": round(statistics.mean(r["tokens"] for r in rows)),
        "rejected_by_guards": sum(r["outcome"] == "rejected" for r in rows),
    }
    categories = sorted({r["category"] for r in rows})
    by_category = {c: round(sum(r["status_ok"] and (r["expect"] != "answered" or r["facts_ok"]) for r in rows if r["category"] == c)
                         / sum(r["category"] == c for r in rows), 2) for c in categories}  # fmt: skip

    print(f"{summary['questions']} questions · p50 {summary['p50_seconds']} s · {summary['tokens_per_question']} tokens each\n")
    print(f"  answered correctly (right facts)   {summary['answered_correctly']:.2f}  of {len(should_answer)} answerable")
    print(f"  cited the right moment             {summary['cited_correctly']:.2f}")
    print(f"  refused an answerable question     {summary['refused_answerable']}")
    print(f"  MADE UP an answer                  {summary['made_up']} of {summary['should_not_answer']} unanswerable")
    print(f"  answers rejected by the guards     {summary['rejected_by_guards']}")
    print("\nall-correct by category: " + " · ".join(f"{c} {v:.2f}" for c, v in by_category.items()))
    print("\nwrong:")
    for r in rows:
        ok = r["status_ok"] and (r["expect"] != "answered" or r["facts_ok"])
        if not ok or (r["expect"] == "answered" and not r["citation_ok"]):
            flag = "" if ok else "✗ "
            print(f"  {flag}[{r['category']}] {r['question']}\n      → {r['status']}: {r['answer'][:120]} {r['notes'] or ''}"
                  f"{'' if r['citation_ok'] or r['expect'] != 'answered' else ' (citation missed)'}")  # fmt: skip

    if args.save:
        Path(args.save).write_text(json.dumps({"summary": summary, "by_category": by_category, "rows": rows}, indent=2) + "\n")
        print(f"\nsaved → {args.save}")


if __name__ == "__main__":
    main()
