"""Generate a spoken talk with known content and upload it: the test material for question answering.

    uv run python scripts/make_talk_video.py eval/talks/sleep_talk.txt      # macOS only (uses the `say` voice)

Real talks would be better test material, but their answers aren't known exactly; here every fact is in the script,
so eval/questions.json can be labelled from it. Synthetic speech is easier to transcribe than a real speaker: a
limitation noted in the results. ffmpeg runs inside the worker container (the Mac may not have it installed).
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

API = "http://localhost:8000/api/v1"
ADMIN = {"x-admin-token": os.environ.get("ADMIN_TOKEN", "dev-only-admin-token-change-me")}  # no limits; library uploads
WORKER = "video-rag-worker"
# Plain backgrounds, one per third of the talk. Avoids the colours the search evaluation asks for (orange, turquoise).
SCENES = ("dimgray", "darkolivegreen", "maroon")


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--voice", default="Samantha")
    args = parser.parse_args()
    if sys.platform != "darwin":
        sys.exit("needs macOS `say`")

    name = args.script.stem
    tmp = Path(tempfile.mkdtemp())
    run(["say", "-v", args.voice, "-o", str(tmp / "speech.aiff"), "-f", str(args.script)])
    run(["docker", "cp", str(tmp / "speech.aiff"), f"{WORKER}:/tmp/{name}.aiff"])
    colours = " ".join(SCENES)
    mux = (
        f"D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 /tmp/{name}.aiff); "
        f'T=$(python3 -c "print(round($D/{len(SCENES)}, 2))"); I=""; F=""; n=0; '
        f'for c in {colours}; do I="$I -f lavfi -i color=c=$c:s=640x360:r=25:d=$T"; F="$F[$n:v]"; n=$((n+1)); done; '
        f"ffmpeg -hide_banner -loglevel error -y $I -i /tmp/{name}.aiff "
        f"-filter_complex \"${{F}}concat=n={len(SCENES)}:v=1[v]\" -map '[v]' -map {len(SCENES)}:a -t $D "
        f"-c:v libx264 -pix_fmt yuv420p -c:a aac /tmp/{name}.mp4 && echo $D"
    )
    duration = run(["docker", "exec", WORKER, "bash", "-c", mux]).strip()
    run(["docker", "cp", f"{WORKER}:/tmp/{name}.mp4", str(tmp / f"{name}.mp4")])
    print(f"generated {name}.mp4 ({float(duration):.1f} s)")

    with open(tmp / f"{name}.mp4", "rb") as fh:
        video = (
            httpx.post(f"{API}/videos", files={"file": (f"{name}.mp4", fh, "video/mp4")}, timeout=300, headers=ADMIN)
            .raise_for_status()
            .json()
        )
    print(f"uploaded → {video['id']}, processing", end="", flush=True)
    for _ in range(120):
        status = httpx.get(f"{API}/videos/{video['id']}", timeout=30, headers=ADMIN).json()
        if status["status"] in ("ready", "failed"):
            break
        print(".", end="", flush=True)
        time.sleep(3)
    print(f" {status['status']}: {status.get('segment_counts')}")


if __name__ == "__main__":
    main()
