"""Print how many tokens the model calls consumed and their estimated cost (from the llm_calls table).

uv run python scripts/usage_report.py [--origin api|worker|eval] [--hours 24] [--save out.json]
"""

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.db.factory import make_database
from src.services.usage.report import usage_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", choices=["api", "worker", "eval"])
    parser.add_argument("--hours", type=float, help="only the last N hours")
    parser.add_argument("--save", help="write the report to this JSON file")
    args = parser.parse_args()

    since = datetime.now(UTC) - timedelta(hours=args.hours) if args.hours else None
    settings = Settings()
    with make_database().get_session() as session:
        report = usage_report(session, since=since, origin=args.origin)

    print(f"price: {settings.llm_price_reference} (${settings.llm_price_input_per_mtok}/M in, "
          f"${settings.llm_price_output_per_mtok}/M out) — an estimate; the model runs locally\n")  # fmt: skip
    header = f"{'':<14}{'calls':>7}{'in tokens':>11}{'out tokens':>12}{'total':>10}{'cost $':>11}"
    print(header)
    for name, t in [("all", report["overall"]), *report["by_operation"].items()]:
        print(
            f"{name:<14}{t['calls']:>7}{t['prompt_tokens']:>11}{t['output_tokens']:>12}{t['total_tokens']:>10}{t['cost_usd']:>11.4f}"
        )
    print("\nper call:")
    for name, t in report["by_operation"].items():
        speed = report["speed"][name]
        print(f"  {name:<12} {t['tokens_per_call']:>6} tokens · ${t['cost_per_1000_calls_usd']:.3f} per 1000 calls · "
              f"reading the prompt {speed['avg_prompt_sec']} s · writing {speed['output_tokens_per_sec']} tokens/s")  # fmt: skip
    if report["understand_outcomes"]:
        print("\nwhat became of the model's answer (understand):")
        for outcome, t in report["understand_outcomes"].items():
            print(f"  {outcome:<10}{t['calls']:>5} calls {t['total_tokens']:>8} tokens")
        print(f"  wasted (rejected + invalid): {report['understand_wasted_share']:.0%} of understanding tokens")
    for name, c in report["cache"].items():
        print(f"\ncache ({name}): {c['hits']} hits · hit rate {c['hit_rate']:.0%} · "
              f"saved {c['saved_tokens']} tokens (${c['saved_cost_usd']:.4f})")  # fmt: skip
    print("\nby origin: " + ", ".join(f"{o} {t['total_tokens']} tokens" for o, t in report["by_origin"].items()))
    if report["videos"]:
        print(f"\ncaptions per video ({len(report['videos'])} videos, most tokens first):")
        for v in report["videos"][:10]:
            print(f"  {(v['title'] or v['video_id'])[:36]:<36}{v['keyframes']:>4} frames {v['total_tokens']:>7} tokens "
                  f"{v['tokens_per_minute'] or '-':>7} /min  ${v['cost_usd']:.4f}")  # fmt: skip

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nsaved → {args.save}")


if __name__ == "__main__":
    main()
