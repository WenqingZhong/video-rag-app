"""Build the image-query evaluation set: eval/images.json (labels, committed) + eval/images/*.jpg (downloaded, ignored).

    uv run python scripts/prepare_image_eval.py

Three kinds of query image:
  near_duplicate  a frame of a library video, taken a quarter into a shot (not the stored keyframe) → that video
  semantic        a Pexels *photo* of a subject the library has → the videos labelled relevant for it in eval/queries.json
  negative        a Pexels photo of a subject the library doesn't have → no clip should come back
Photos are the first results Pexels returns for each subject (not hand-picked). Re-running re-downloads the images
listed in eval/images.json if it exists, so labels stay fixed.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text

from src.config import Settings
from src.services.storage import make_storage_client

LABELS = ROOT / "eval" / "images.json"
IMAGES = ROOT / "eval" / "images"
WORKER = "video-rag-worker"
PER_SUBJECT = 2

# photo search → the eval/queries.json query whose relevant videos it should find
SEMANTIC = {
    "dog": "give me a clip of a dog",
    "kitten": "a kitten resting",
    "ocean waves": "ocean waves",
    "sandy beach": "sandy beach",
    "colorful parrot": "a colourful parrot-like bird",
    "city traffic at night": "busy city streets at night",
    "cooking in a pan": "cooking food in a pan",
    "business meeting": "people having a meeting",
    "sunlight forest": "sunlight in the woods",
    "car dashboard navigation": "car dashboard with navigation",
}
# absent from the library (the same subjects as eval/no_answer.json, plus two)
NEGATIVE = ["elephant", "piano", "horse", "football match", "snowboarder", "rocket launch", "giraffe", "violin"]


def pexels_photos(settings: Settings, query: str) -> list[dict]:
    response = httpx.get(
        "https://api.pexels.com/v1/search",
        params={"query": query, "per_page": PER_SUBJECT, "orientation": "landscape"},
        headers={"Authorization": settings.pexels_api_key},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["photos"]


def download(url: str, path: Path) -> None:
    if not path.exists():
        path.write_bytes(httpx.get(url, timeout=60, follow_redirects=True).raise_for_status().content)


def near_duplicates(settings: Settings, count: int = 10) -> list[dict]:
    """A frame a quarter into one shot of each of `count` Pexels videos: similar to, but not, the stored keyframe."""
    engine = create_engine(settings.postgres_database_url)
    with engine.connect() as c:
        rows = list(
            c.execute(
                text(
                    """SELECT DISTINCT ON (v.id) v.id, v.source_id, v.s3_key, s.start_sec, s.end_sec, s.frame_time_sec
                       FROM videos v JOIN segments s ON s.video_id = v.id AND s.kind = 'visual'
                       WHERE v.status = 'ready' AND v.source = 'pexels' AND s.end_sec - s.start_sec >= 2
                       ORDER BY v.id, s.idx"""
                )
            )
        )[:count]
    storage = make_storage_client(settings)
    items = []
    with tempfile.TemporaryDirectory() as tmp:
        for r in rows:
            at = round(r.start_sec + (r.end_sec - r.start_sec) / 4, 2)
            name = f"near_{r.source_id}.jpg"
            if not (IMAGES / name).exists():
                source = storage.download_file(r.s3_key, Path(tmp) / "source.mp4")
                subprocess.run(["docker", "cp", str(source), f"{WORKER}:/tmp/eval_source.mp4"], check=True)
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        WORKER,
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        str(at),
                        "-i",
                        "/tmp/eval_source.mp4",
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=640:-2",
                        f"/tmp/{name}",
                    ],
                    check=True,
                )
                subprocess.run(["docker", "cp", f"{WORKER}:/tmp/{name}", str(IMAGES / name)], check=True)
            items.append(
                {
                    "file": name,
                    "kind": "near_duplicate",
                    "relevant": [r.source_id],
                    "note": f"frame at {at}s; keyframe at {r.frame_time_sec}s",
                }
            )
    return items


def main() -> None:
    settings = Settings()
    IMAGES.mkdir(parents=True, exist_ok=True)
    if LABELS.exists():  # labels are fixed: only fetch missing files
        labelled = json.loads(LABELS.read_text())["images"]
        for item in labelled:
            if item.get("url"):
                download(item["url"], IMAGES / item["file"])
        near_duplicates(settings)
        print(f"{len(labelled)} labelled images present in {IMAGES}")
        return

    queries = {q["query"]: q["relevant"] for q in json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]}
    items = near_duplicates(settings)
    for subject, query in SEMANTIC.items():
        for photo in pexels_photos(settings, subject):
            name = f"photo_{photo['id']}.jpg"
            download(photo["src"]["medium"], IMAGES / name)
            items.append(
                {
                    "file": name,
                    "kind": "semantic",
                    "subject": subject,
                    "relevant": queries[query],
                    "url": photo["src"]["medium"],
                    "credit": f"{photo['photographer']} on Pexels",
                }
            )
    for subject in NEGATIVE:
        for photo in pexels_photos(settings, subject):
            name = f"photo_{photo['id']}.jpg"
            download(photo["src"]["medium"], IMAGES / name)
            items.append(
                {
                    "file": name,
                    "kind": "negative",
                    "subject": subject,
                    "relevant": [],
                    "url": photo["src"]["medium"],
                    "credit": f"{photo['photographer']} on Pexels",
                }
            )

    description = (
        "Image queries: near_duplicate (a non-keyframe frame of a library video), semantic (a Pexels photo of a subject "
        "the library has; relevant = eval/queries.json labels), negative (a subject the library lacks: no clip expected)."
    )
    LABELS.write_text(json.dumps({"description": description, "images": items}, indent=2) + "\n")
    counts = {k: sum(1 for i in items if i["kind"] == k) for k in ("near_duplicate", "semantic", "negative")}
    print(f"wrote {LABELS}: {counts}")


if __name__ == "__main__":
    main()
