"""What the caches save, measured on the running API with the evaluation requests (48: 36 search + 12 no-answer).

    make eval-cache      # = uv run python scripts/evaluate_cache.py --save eval/results/cache.json

Clears only the understanding and answer caches (not clips), then sends the requests in passes:
  1. cold        every request once: nothing cached
  2. repeat      the same requests again, re-typed in CAPITALS: answer cache
  3. new filter  the same requests with max_clips=2: the answer is new, the understanding is not
  4. index bump  one video re-indexed, then a few requests again: cached answers must be recomputed
Checks that cached answers are identical to fresh ones, so the speed-up doesn't change results.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import redis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Settings

API = "http://localhost:8000/api/v1"


def clear_caches(settings: Settings) -> int:
    client = redis.Redis.from_url(settings.redis_url)
    keys = [k for pattern in ("understand:*", "answer:*") for k in client.scan_iter(pattern, count=1000)]
    return client.delete(*keys) if keys else 0


def ask(http: httpx.Client, query: str, **extra) -> dict:
    started = time.perf_counter()
    body = http.post(f"{API}/ask", json={"query": query, **extra}).raise_for_status().json()
    body["_seconds"] = time.perf_counter() - started
    return body


def fingerprint(body: dict) -> tuple:
    """What the user gets: status, answer text and which clips. A cached answer must match exactly."""
    return body["status"], body["answer"], tuple((c["video"]["id"], c["start_sec"], c["end_sec"]) for c in body["clips"])


def summarize(name: str, bodies: list[dict]) -> dict:
    seconds = sorted(b["_seconds"] for b in bodies)
    usage = [b["usage"] for b in bodies]
    count = lambda part, state: sum(1 for b in bodies if b["cache"].get(part) == state)
    return {
        "pass": name,
        "requests": len(bodies),
        "p50_ms": round(1000 * statistics.median(seconds)),
        "p95_ms": round(1000 * seconds[int(0.95 * (len(seconds) - 1))]),
        "answer_hits": count("answer", "hit"),
        "understanding_hits": count("understanding", "hit"),
        "tokens_spent": sum(u["total_tokens"] for u in usage),
        "tokens_saved": sum(u["saved_tokens"] for u in usage),
        "cost_spent_usd": round(sum(u["cost_usd"] for u in usage), 6),
        "cost_saved_usd": round(sum(u["saved_cost_usd"] for u in usage), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    settings = Settings()
    queries = [q["query"] for q in json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]]
    queries += json.loads((ROOT / "eval" / "no_answer.json").read_text())["queries"]
    print(f"cleared {clear_caches(settings)} cache entries · {len(queries)} requests\n")

    http = httpx.Client(timeout=120)
    ask(http, "warm up the model")  # so pass 1 isn't charged for loading the model
    clear_caches(settings)

    cold = [ask(http, q) for q in queries]
    repeat = [ask(http, q.upper()) for q in queries]
    new_filter = [ask(http, q, max_clips=2) for q in queries]

    # Re-index one video that appears in an answer: the index version goes up, cached answers must not be served.
    video_id = next(c["video"]["id"] for b in cold for c in b["clips"])
    http.post(f"{API}/admin/reindex", json={"video_id": video_id}).raise_for_status()
    time.sleep(3)  # the worker re-indexes and bumps the version
    sample = queries[:10]
    after_bump = [ask(http, q) for q in sample]

    passes = [
        summarize("1 cold", cold),
        summarize("2 repeat (answer cache)", repeat),
        summarize("3 max_clips=2 (understanding cache)", new_filter),
        summarize("4 after re-index", after_bump),
    ]
    identical = sum(fingerprint(a) == fingerprint(b) for a, b in zip(cold, repeat, strict=True))
    identical_after_bump = sum(fingerprint(a) == fingerprint(b) for a, b in zip(cold[:10], after_bump, strict=True))

    print(f"{'pass':<38}{'p50 ms':>8}{'p95 ms':>8}{'ans hit':>9}{'und hit':>9}{'spent':>8}{'saved':>8}")
    for p in passes:
        print(
            f"{p['pass']:<38}{p['p50_ms']:>8}{p['p95_ms']:>8}{p['answer_hits']:>9}{p['understanding_hits']:>9}"
            f"{p['tokens_spent']:>8}{p['tokens_saved']:>8}"
        )
    print(f"\ncached answers identical to fresh ones: {identical}/{len(queries)}")
    print(f"after re-index: {passes[3]['answer_hits']} answer hits (expected 0), {identical_after_bump}/10 same as cold")

    if args.save:
        report = {
            "requests": len(queries),
            "passes": passes,
            "identical": identical,
            "identical_after_bump": identical_after_bump,
        }
        Path(args.save).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nsaved → {args.save}")


if __name__ == "__main__":
    main()
