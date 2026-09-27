"""Show where the time went: recent (or slowest) traces, or one trace as a timeline.

uv run python scripts/show_trace.py                      # the 15 most recent requests and tasks
uv run python scripts/show_trace.py --slowest --name "POST /api/v1/ask"
uv run python scripts/show_trace.py <trace id>           # one trace, API and workers together
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.factory import make_database
from src.services.tracing import get_trace, list_traces, waterfall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_id", nargs="?")
    parser.add_argument("--name", help='root span name, e.g. "POST /api/v1/ask"')
    parser.add_argument("--slowest", action="store_true")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    with make_database().get_session() as session:
        if args.trace_id:
            trace = get_trace(session, args.trace_id)
            if trace is None:
                sys.exit(f"no trace {args.trace_id}")
            llm = trace["llm"]
            print(f"trace {args.trace_id} · {llm['calls']} model calls · {llm['tokens']} tokens · ${llm['cost_usd']:.6f}\n")
            print(waterfall(trace))
            return
        for t in list_traces(session, name=args.name, slowest=args.slowest, limit=args.limit):
            flag = " ✗" if t["status"] == "error" else ""
            print(f"{t['started_at']:%H:%M:%S}  {t['trace_id']}  {t['duration_ms']:>9.1f} ms  {t['name']}{flag}")


if __name__ == "__main__":
    main()
