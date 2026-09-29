"""Image queries: does a photo find the right video, and does an unrelated photo find nothing?

    make eval-images      # = uv run python scripts/evaluate_images.py --save eval/results/images.json

Needs eval/images/ (scripts/prepare_image_eval.py). Each image is embedded once (CLIP, as the API does); its nearest
keyframes are fetched once; then the similarity cut-off is swept offline:
  hit@1           the best video is a relevant one (and passes the cut-off)
  false answers   a negative image (subject not in the library) still gets a clip
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text

from src.config import Settings
from src.services.embeddings import make_embedding_client
from src.services.opensearch import make_opensearch_service
from src.services.search import query_builder as qb

CUTOFFS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    settings = Settings()
    labelled = json.loads((ROOT / "eval" / "images.json").read_text())["images"]
    embedder, opensearch = make_embedding_client(settings), make_opensearch_service(settings)
    engine = create_engine(settings.postgres_database_url)
    with engine.connect() as c:
        pexels_id = {r.id: r.source_id or r.title for r in c.execute(text("SELECT id, source_id, title FROM videos"))}

    rows = []
    filters = qb.build_filters(kind="visual")  # the shared library, as any visitor sees it
    for item in labelled:
        vector = embedder.embed_images([(ROOT / "eval" / "images" / item["file"]).read_bytes()])[0]
        raw = opensearch.search(qb.vector_query(vector, filters, 20))["hits"]["hits"]
        best: dict[str, float] = {}  # video → best keyframe similarity
        for r in raw:
            video = pexels_id.get(r["_source"]["video_id"], r["_source"]["video_id"])
            best[video] = max(best.get(video, -1), 2 * float(r["_score"]) - 1)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        top_video, top_similarity = ranked[0] if ranked else (None, -1.0)
        rows.append({**item, "top_video": top_video, "top_similarity": round(top_similarity, 4),
                     "top_relevant": top_video in item["relevant"]})  # fmt: skip

    def score(cutoff: float) -> dict:
        out = {"cutoff": cutoff}
        for kind in ("near_duplicate", "semantic"):
            group = [r for r in rows if r["kind"] == kind]
            out[f"{kind}_hit@1"] = round(sum(r["top_relevant"] and r["top_similarity"] >= cutoff for r in group) / len(group), 3)
        negatives = [r for r in rows if r["kind"] == "negative"]
        out["false_answers"] = sum(r["top_similarity"] >= cutoff for r in negatives)
        out["negatives"] = len(negatives)
        return out

    sweep = [score(c) for c in CUTOFFS]
    print(
        f"{len(rows)} images: "
        + ", ".join(f"{k} {sum(r['kind'] == k for r in rows)}" for k in ("near_duplicate", "semantic", "negative"))
    )
    print("\nsimilarity of the best keyframe, by kind (min / median / max):")
    for kind in ("near_duplicate", "semantic", "negative"):
        sims = sorted(r["top_similarity"] for r in rows if r["kind"] == kind)
        print(f"  {kind:<15} {sims[0]:.3f} / {sims[len(sims) // 2]:.3f} / {sims[-1]:.3f}")
    print(f"\n{'cut-off':>8}{'near-dup hit@1':>16}{'semantic hit@1':>16}{'false answers':>15}")
    for s in sweep:
        print(
            f"{s['cutoff']:>8.2f}{s['near_duplicate_hit@1']:>16.2f}{s['semantic_hit@1']:>16.2f}{s['false_answers']:>9} / {s['negatives']}"
        )

    print("\nsemantic misses (top video not relevant) and the strongest negatives:")
    for r in rows:
        if r["kind"] == "semantic" and not r["top_relevant"]:
            print(f"  semantic {r['subject']:<26} top {r['top_video']} ({r['top_similarity']:.3f}) not in {r['relevant']}")
    for r in sorted((r for r in rows if r["kind"] == "negative"), key=lambda r: -r["top_similarity"])[:5]:
        print(f"  negative {r['subject']:<26} top {r['top_video']} ({r['top_similarity']:.3f})")

    if args.save:
        Path(args.save).write_text(json.dumps({"sweep": sweep, "rows": rows}, indent=2) + "\n")
        print(f"\nsaved → {args.save}")


if __name__ == "__main__":
    main()
