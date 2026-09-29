"""Whole conversations through the running API: does the agent get there, turn by turn?

    make eval-chat      # = uv run python scripts/evaluate_conversations.py --save eval/results/conversations.json

For each conversation in eval/conversations.json, sends the turns to POST /api/v1/chat (one conversation id) and checks
each turn's action, clip and reply against its labels. Reports turn and conversation success, time per turn, and
tokens per turn (from the turn's trace). Every conversation starts fresh; nothing is downloaded from Pexels.
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
API = "http://localhost:8000/api/v1"


def check(turn: dict, body: dict, previous_video: str | None) -> list[str]:
    expect, problems = turn["expect"], []
    if body["action"] not in expect["action"]:
        problems.append(f"action {body['action']} (expected {'/'.join(expect['action'])})")
    clip = body["clips"][0] if body["clips"] else None
    if "clip_title" in expect:
        title = (clip or {}).get("title") or ""
        if not clip or not re.search(expect["clip_title"], title, re.IGNORECASE):
            problems.append(f"clip {title or 'none'!r} doesn't match /{expect['clip_title']}/")
    if expect.get("new_video") and clip and clip.get("video_id") == previous_video:
        problems.append("same video as the previous turn")
    if "reply_has" in expect and not re.search(expect["reply_has"], body["reply"], re.IGNORECASE):
        problems.append(f"reply lacks /{expect['reply_has']}/")
    if "reply_not" in expect and re.search(expect["reply_not"], body["reply"], re.IGNORECASE):
        problems.append(f"reply has /{expect['reply_not']}/")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    conversations = json.loads((ROOT / "eval" / "conversations.json").read_text())["conversations"]
    images = {i.get("subject"): i["file"] for i in json.loads((ROOT / "eval" / "images.json").read_text())["images"]}
    http = httpx.Client(timeout=300)
    results = []
    for conversation in conversations:
        cid, previous_video, turns = f"eval-{int(time.time() * 1000)}", None, []
        for turn in conversation["turns"]:
            files = None
            if turn.get("image"):
                files = {"image": ("photo.jpg", (ROOT / "eval" / "images" / images[turn["image"]]).read_bytes(), "image/jpeg")}
            started = time.perf_counter()
            body = (
                http.post(f"{API}/chat", data={"message": turn["message"], "conversation_id": cid}, files=files)
                .raise_for_status()
                .json()
            )
            seconds = time.perf_counter() - started
            time.sleep(0.5)  # the trace is saved right after the response
            trace = http.get(f"{API}/traces/{body['request_id']}").json() if body.get("request_id") else {}
            problems = check(turn, body, previous_video)
            turns.append({"message": turn["message"], "action": body["action"], "decided_by": body["decided_by"],
                          "reply": body["reply"][:200], "clip": (body["clips"][0].get("title") if body["clips"] else None),
                          "ok": not problems, "problems": problems, "seconds": round(seconds, 2),
                          "tokens": trace.get("llm", {}).get("tokens", 0), "notes": body["notes"]})  # fmt: skip
            if body["clips"]:
                previous_video = body["clips"][0].get("video_id")
        results.append({"name": conversation["name"], "ok": all(t["ok"] for t in turns), "turns": turns})
        mark = "✓" if results[-1]["ok"] else "✗"
        print(f"{mark} {conversation['name']}", flush=True)
        for t in turns:
            print(f"    {'·' if t['ok'] else '✗'} {t['message']!r:<46} {t['action']:<16} {t['decided_by']:<5} {t['seconds']:>5.1f} s"
                  f"  {'; '.join(t['problems'])}")  # fmt: skip

    all_turns = [t for r in results for t in r["turns"]]
    summary = {
        "conversations": len(results),
        "conversations_ok": sum(r["ok"] for r in results),
        "turns": len(all_turns),
        "turns_ok": sum(t["ok"] for t in all_turns),
        "p50_seconds": round(statistics.median(t["seconds"] for t in all_turns), 2),
        "p95_seconds": round(sorted(t["seconds"] for t in all_turns)[int(0.95 * (len(all_turns) - 1))], 2),
        "tokens_per_turn": round(statistics.mean(t["tokens"] for t in all_turns)),
        "decided_by_rules": sum(t["decided_by"] == "rules" for t in all_turns),
    }
    print(
        f"\nconversations fully right: {summary['conversations_ok']}/{summary['conversations']} · "
        f"turns right: {summary['turns_ok']}/{summary['turns']} · p50 {summary['p50_seconds']} s, p95 {summary['p95_seconds']} s · "
        f"{summary['tokens_per_turn']} tokens/turn · rules decided {summary['decided_by_rules']}/{summary['turns']}"
    )
    if args.save:
        Path(args.save).write_text(json.dumps({"summary": summary, "conversations": results}, indent=2) + "\n")
        print(f"saved → {args.save}")
    sys.exit(0)


if __name__ == "__main__":
    main()
