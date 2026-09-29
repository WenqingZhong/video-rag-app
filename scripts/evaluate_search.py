"""Measure search quality: keyword vs vector vs hybrid on a labelled query set.

    uv run python scripts/evaluate_search.py                 # uses eval/queries.json, API at localhost:8000
    uv run python scripts/evaluate_search.py --k 5 --json    # machine-readable output

Relevance is judged per VIDEO (results are grouped by video), labelled by Pexels id so the
labels stay valid across environments (our internal video ids differ per database).

Metrics (averaged over queries):
- Recall@k: share of a query's relevant videos in the top k, out of min(k, #relevant)   ("did we find them?")
            (capped so a query with 9 relevant videos can still score 1.0 at k=5)
- MRR:      1 / rank of the FIRST relevant video (0 if none in the results)     ("how soon?")
"""

import argparse
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MODES = ["keyword", "vector", "hybrid", "understood"]  # understood = the LLM parses the request first (ADR 0002)


def search(api: str, query: str, mode: str, size: int) -> list[str]:
    response = httpx.post(
        f"{api}/api/v1/search",
        json={
            "query": query,
            "mode": "auto" if mode == "understood" else mode,
            "understand": mode == "understood",
            "size": size,
            "group_by_video": True,
            "source": "pexels",
        },
        timeout=60,
    )
    response.raise_for_status()
    return [hit["video"]["id"] for hit in response.json()["hits"]]


def pexels_ids(api: str) -> dict[str, str]:
    """internal video id → Pexels id."""
    videos = httpx.get(f"{api}/api/v1/videos", params={"source": "pexels", "limit": 200}, timeout=30).json()["items"]
    return {v["id"]: v["source_id"] for v in videos}


def evaluate(api: str, queries: list[dict], k: int) -> dict:
    to_pexels = pexels_ids(api)
    report: dict = {"k": k, "modes": {}, "queries": []}
    per_query = {q["query"]: {} for q in queries}
    for mode in MODES:
        recalls, reciprocal_ranks = [], []
        for q in queries:
            relevant = set(q["relevant"])
            ranked = [to_pexels.get(video_id) for video_id in search(api, q["query"], mode, size=max(k, 10))]
            found = relevant & set(ranked[:k])
            recall = len(found) / min(k, len(relevant))
            first = next((i for i, pid in enumerate(ranked, 1) if pid in relevant), None)
            rr = 1 / first if first else 0.0
            recalls.append(recall)
            reciprocal_ranks.append(rr)
            per_query[q["query"]][mode] = {"recall": round(recall, 2), "first_relevant_rank": first}
        report["modes"][mode] = {
            f"recall@{k}": round(sum(recalls) / len(recalls), 3),
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
        }
    report["queries"] = [{"query": q["query"], "category": q.get("category", "basic"), **per_query[q["query"]]} for q in queries]
    categories = sorted({row["category"] for row in report["queries"]}, key=lambda c: (c != "basic", c))
    report["categories"] = {
        c: {
            mode: round(
                sum(1 / r[mode]["first_relevant_rank"] if r[mode]["first_relevant_rank"] else 0.0 for r in rows) / len(rows), 3
            )
            for mode in MODES
        }
        for c in categories
        for rows in [[r for r in report["queries"] if r["category"] == c]]
    }
    return report


def false_answers(api: str, queries: list[str]) -> dict:
    """Requests whose correct answer is NO clip: how many does each mode answer anyway?"""
    report = {}
    for mode in MODES:
        answered = [q for q in queries if search(api, q, mode, size=1)]
        report[mode] = {"false_answers": len(answered), "of": len(queries), "answered": answered}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--queries", default=str(ROOT / "eval" / "queries.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save", help="also write the full report to this JSON file")
    args = parser.parse_args()

    queries = json.loads(Path(args.queries).read_text())["queries"]
    report = evaluate(args.api, queries, args.k)
    no_answer = json.loads((ROOT / "eval" / "no_answer.json").read_text())["queries"]
    report["false_answers"] = false_answers(args.api, no_answer)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"{len(queries)} queries, relevance per video, k={args.k}\n")
    print(f"{'mode':<9}{'Recall@' + str(args.k):>10}{'MRR':>8}")
    for mode, metrics in report["modes"].items():
        print(f"{mode:<9}{metrics[f'recall@{args.k}']:>10.3f}{metrics['mrr']:>8.3f}")
    print("\nMRR per category:")
    print(f"{'category':<10}{'queries':>8}" + "".join(f"{m:>10}" for m in MODES))
    for category, scores in report["categories"].items():
        count = sum(1 for r in report["queries"] if r["category"] == category)
        print(f"{category:<10}{count:>8}" + "".join(f"{scores[m]:>10.2f}" for m in MODES))
    print(f"\nfalse answers (should return nothing, {len(no_answer)} requests in eval/no_answer.json):")
    for mode, fa in report["false_answers"].items():
        print(f"{mode:<11}{fa['false_answers']:>3} / {fa['of']}   {', '.join(repr(q) for q in fa['answered'][:6])}")
    print("\nper query (recall / rank of first relevant):")
    print(f"{'query':<34}" + "".join(f"{m:>16}" for m in MODES))
    for row in report["queries"]:
        cells = "".join(f"{row[m]['recall']:>9.2f} / {row[m]['first_relevant_rank'] or '-'!s:<4}" for m in MODES)
        print(f"{row['query'][:33]:<34}{cells}")


if __name__ == "__main__":
    main()
