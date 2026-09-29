"""Can the agent pick the right action for a chat turn? The model alone vs model + guards vs rules alone.

    make eval-agent      # = uv run python scripts/evaluate_agent_turns.py --save eval/results/agent_turns.json

For each labelled turn in eval/agent_turns.json (context + message → expected action):
  action      the right tool (or a plain reply)
  text        the request/question it passes on has the right words (a follow-up combined with the last request)
  all         action and text both right
  unsafe      a Pexels download without the user's consent (must be 0)
Needs Ollama running with the model.
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.services.agent.decide import Decision, LLMRouter, TurnContext, decide, decide_with_rules
from src.services.llm import make_chat_model
from src.services.understanding import find_exclusions
from src.services.understanding.llm import InvalidReply


def model_only(router: LLMRouter):
    def run(message: str, context: TurnContext) -> Decision:
        try:
            reply, call = router.route(message, context)
        except InvalidReply as exc:
            return Decision("reply", source="llm", notes=[f"invalid: {exc}"], llm_call=exc.call)
        return Decision(reply.action, text=reply.text, about_last_video=reply.about_last_video, source="llm", llm_call=call)

    return run


def score(decision: Decision, expect: dict) -> dict:
    text = (decision.text or "").lower()
    action_ok = decision.action == expect["action"]
    text_ok = action_ok
    if action_ok and decision.action in ("find_clip", "answer_question", "fetch_from_pexels"):
        stems = lambda words: [w.lower().rstrip("s") for w in words]
        text_ok = all(w in text for w in stems(expect.get("text_has", []))) and not any(
            w in text for w in stems(expect.get("text_not", []))
        )
        if expect.get("exclude"):
            text_ok = text_ok and set(expect["exclude"]) <= set(find_exclusions(decision.text or ""))
        if "same_video" in expect:
            text_ok = text_ok and decision.about_last_video == expect["same_video"]
    return {"action": action_ok, "text": text_ok, "all": action_ok and text_ok}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    parser.add_argument("--turns", default="eval/agent_turns.json")
    args = parser.parse_args()

    settings = Settings()
    turns = json.loads((ROOT / args.turns).read_text())["turns"]
    router = LLMRouter(make_chat_model(settings, "text"))
    router.route("hi", TurnContext())  # load the model first
    systems = {
        "rules only": lambda m, c: decide_with_rules(m, c),
        "model only": model_only(router),
        "rules first + model": lambda m, c: decide(m, c, router),
    }

    reports = {}
    for name, system in systems.items():
        rows, seconds = [], []
        for turn in turns:
            context = TurnContext(**turn["context"])
            started = time.perf_counter()
            decision = system(turn["message"], context)
            seconds.append(time.perf_counter() - started)
            unsafe = decision.action == "fetch_from_pexels" and turn["expect"]["action"] != "fetch_from_pexels"
            rows.append({"category": turn["category"], "message": turn["message"], "expect": turn["expect"],
                         "got": {"action": decision.action, "text": decision.text, "about_last_video": decision.about_last_video},
                         "source": decision.source, "notes": decision.notes, "unsafe": unsafe,
                         "tokens": decision.llm_call.total_tokens if decision.llm_call else 0,
                         **score(decision, turn["expect"])})  # fmt: skip
        by_category = defaultdict(list)
        for r in rows:
            by_category[r["category"]].append(r["all"])
        reports[name] = {
            "action": round(statistics.mean(r["action"] for r in rows), 3),
            "all": round(statistics.mean(r["all"] for r in rows), 3),
            "unsafe": sum(r["unsafe"] for r in rows),
            "model_calls": round(statistics.mean(r["tokens"] > 0 for r in rows), 2),
            "p50_seconds": round(statistics.median(seconds), 2),
            "tokens_per_turn": round(statistics.mean(r["tokens"] for r in rows)),
            "by_category": {c: round(statistics.mean(v), 2) for c, v in by_category.items()},
            "rows": rows,
        }

    print(f"{len(turns)} labelled turns\n")
    print(f"{'system':<26}{'action':>8}{'all':>7}{'unsafe':>8}{'p50 s':>8}{'tokens':>8}")
    for name, rep in reports.items():
        print(
            f"{name:<26}{rep['action']:>8.2f}{rep['all']:>7.2f}{rep['unsafe']:>8}{rep['p50_seconds']:>8.2f}{rep['tokens_per_turn']:>8}"
        )
    categories = list(reports["rules only"]["by_category"])
    print("\nall-correct by category:")
    print(f"{'category':<16}{'n':>4}" + "".join(f"{name:>26}" for name in reports))
    for c in categories:
        n = sum(1 for t in turns if t["category"] == c)
        print(f"{c:<16}{n:>4}" + "".join(f"{rep['by_category'][c]:>26.2f}" for rep in reports.values()))
    for name in ("model only", "rules first + model"):
        print(f"\n{name}: wrong turns")
        for r in reports[name]["rows"]:
            if not r["all"]:
                g = r["got"]
                print(
                    f"  [{r['category']}] {r['message']!r:<48} → {g['action']} {g['text']!r} ({r['source']}) {r['notes'][-1] if r['notes'] else ''}"
                )

    if args.save:
        Path(args.save).write_text(json.dumps(reports, indent=2) + "\n")
        print(f"\nsaved → {args.save}")


if __name__ == "__main__":
    main()
