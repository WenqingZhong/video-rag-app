"""Experiment: does embedding the CAPTIONS (text → vector) improve visual search, next to keyword + CLIP?

Runs offline against the live data, without changing the system:

    uv run --with sentence-transformers python scripts/experiment_caption_embeddings.py [--save eval/results/x.json]
    make experiment-captions      # same, saving to eval/results/caption_embeddings.json

Decision and discussion: docs/decisions/0001-no-caption-embeddings.md

Retrievers (each returns its top 30 segments, like the API's over-fetch):
    keyword       BM25 over transcript + caption + title      (live API, mode=keyword)
    clip_image    CLIP text → keyframe image vectors         (live API, mode=vector)
    caption_bge   bge-small-en-v1.5: query → caption vectors  (computed here)
    caption_clip  CLIP text encoder: query → caption vectors  (embedding service, text-to-text)
Fusions use the same Reciprocal Rank Fusion as the API, then keep each video's best segment.
Relevance and metrics are the same as scripts/evaluate_search.py (per video, Recall@5, MRR).
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.services.search.fusion import reciprocal_rank_fusion
from src.services.search.visual_query import visual_query_text

API = "http://localhost:8000/api/v1"
ADMIN = {"x-admin-token": os.environ.get("ADMIN_TOKEN", "dev-only-admin-token-change-me")}  # no limits; library uploads
TOP = 30
K = 5
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge v1.5 convention for queries


def api_ranking(query: str, mode: str) -> list[str]:
    body = {"query": query, "mode": mode, "size": TOP, "source": "pexels"}
    hits = httpx.post(f"{API}/search", json=body, timeout=60, headers=ADMIN).json()["hits"]
    return [h["segment_id"] for h in hits]


def cosine_ranking(query_vec: list[float], vectors: dict[str, list[float]]) -> list[str]:
    scored = sorted(vectors, key=lambda sid: -sum(a * b for a, b in zip(query_vec, vectors[sid], strict=True)))
    return scored[:TOP]


def to_videos(segment_ids: list[str], segment_video: dict[str, str]) -> list[str]:
    """Segment ranking → video ranking (best segment per video), like group_by_video."""
    seen, videos = set(), []
    for sid in segment_ids:
        vid = segment_video.get(sid)
        if vid and vid not in seen:
            seen.add(vid)
            videos.append(vid)
    return videos


def score(ranked_videos: list[str], relevant: set[str]) -> tuple[float, int | None]:
    recall = len(relevant & set(ranked_videos[:K])) / min(K, len(relevant))
    first = next((i for i, v in enumerate(ranked_videos, 1) if v in relevant), None)
    return recall, first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="write the full results (per system, category and query) to this JSON file")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    settings = Settings()
    with create_engine(settings.postgres_database_url).connect() as conn:
        rows = conn.execute(
            text("""SELECT s.id, v.source_id, s.caption FROM segments s JOIN videos v ON v.id = s.video_id
                    WHERE v.source = 'pexels' AND v.status = 'ready' AND s.kind = 'visual' AND s.caption IS NOT NULL""")
        ).all()
        all_segments = dict(
            conn.execute(text("SELECT s.id, v.source_id FROM segments s JOIN videos v ON v.id = s.video_id")).all()
        )
    captions = {sid: cap for sid, _, cap in rows}
    print(f"{len(captions)} captioned keyframes from {len({src for _, src, _ in rows})} Pexels videos")

    bge = SentenceTransformer("BAAI/bge-small-en-v1.5")
    caption_bge = dict(zip(captions, bge.encode(list(captions.values()), normalize_embeddings=True).tolist(), strict=True))
    clip_text = httpx.post(f"{settings.embedder_url}/embed/text", json={"texts": list(captions.values())}, timeout=120).json()[
        "vectors"
    ]
    caption_clip = dict(zip(captions, clip_text, strict=True))

    queries = json.loads((ROOT / "eval" / "queries.json").read_text())["queries"]
    systems = {
        "keyword": ["keyword"],
        "clip_image": ["clip_image"],
        "caption_bge": ["caption_bge"],
        "caption_clip": ["caption_clip"],
        "keyword+clip (current hybrid)": ["keyword", "clip_image"],
        "keyword+caption_bge": ["keyword", "caption_bge"],
        "clip+caption_bge": ["clip_image", "caption_bge"],
        "keyword+clip+caption_bge": ["keyword", "clip_image", "caption_bge"],
        "keyword+clip+caption_clip": ["keyword", "clip_image", "caption_clip"],
    }
    per_query = []
    for q in queries:
        visual = visual_query_text(q["query"])
        rankings = {
            "keyword": api_ranking(q["query"], "keyword"),
            "clip_image": api_ranking(q["query"], "vector"),
            "caption_bge": cosine_ranking(bge.encode(BGE_QUERY_PREFIX + visual, normalize_embeddings=True).tolist(), caption_bge),
            "caption_clip": cosine_ranking(
                httpx.post(f"{settings.embedder_url}/embed/text", json={"texts": [visual]}).json()["vectors"][0], caption_clip
            ),
        }
        relevant = set(q["relevant"])
        row = {"query": q["query"]}
        for name, parts in systems.items():
            fused = list(reciprocal_rank_fusion({p: rankings[p] for p in parts}, k=60)) if len(parts) > 1 else rankings[parts[0]]
            row[name] = score(to_videos(fused, all_segments), relevant)
        per_query.append(row)

    n = len(queries)
    categories = sorted({q.get("category", "basic") for q in queries}, key=lambda c: (c != "basic", c))
    by_cat = {c: [r for r, q in zip(per_query, queries, strict=True) if q.get("category", "basic") == c] for c in categories}
    hard = [r for r, q in zip(per_query, queries, strict=True) if q.get("category", "basic") != "basic"]

    def mrr(rows, name):
        return sum(1 / r[name][1] if r[name][1] else 0.0 for r in rows) / len(rows)

    def recall(rows, name):
        return sum(r[name][0] for r in rows) / len(rows)

    print(f"\n{n} queries ({len(by_cat['basic'])} basic + {len(hard)} hard), per-video relevance, k={K}\n")
    print(f"{'system':<32}{'all R@5':>9}{'all MRR':>9}{'hard R@5':>10}{'hard MRR':>10}")
    for name in systems:
        print(
            f"{name:<32}{recall(per_query, name):>9.3f}{mrr(per_query, name):>9.3f}{recall(hard, name):>10.3f}{mrr(hard, name):>10.3f}"
        )

    print("\nMRR per category:")
    print(f"{'system':<32}" + "".join(f"{c[:8]:>9}" for c in categories))
    for name in systems:
        print(f"{name:<32}" + "".join(f"{mrr(by_cat[c], name):>9.2f}" for c in categories))

    focus = ["keyword", "clip_image", "caption_bge", "keyword+clip (current hybrid)", "keyword+clip+caption_bge"]
    labels = ["keyword", "clip", "cap_bge", "hybrid", "hybrid+cap"]
    print("\nhard queries: rank of first relevant video (- = not in results)")
    print(f"{'query':<40}{'cat':<9}" + "".join(f"{lbl:>11}" for lbl in labels))
    for r, q in zip(per_query, queries, strict=True):
        if q.get("category", "basic") != "basic":
            print(f"{r['query'][:39]:<40}{q['category']:<9}" + "".join(f"{r[f][1] or '-'!s:>11}" for f in focus))

    if args.save:
        clip_model = httpx.get(f"{settings.embedder_url}/health").json()["model"]
        report = {
            "experiment": "caption embeddings as a third retriever (docs/decisions/0001-no-caption-embeddings.md)",
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "library": {"pexels_videos": len({src for _, src, _ in rows}), "captioned_keyframes": len(captions)},
            "models": {"clip": clip_model, "caption": settings.caption_model, "caption_text_embedding": "BAAI/bge-small-en-v1.5"},
            "method": {"k": K, "retriever_depth": TOP, "fusion": "RRF, k=60", "relevance": "per video (Pexels id)"},
            "queries": {"total": n, "by_category": {c: len(by_cat[c]) for c in categories}},
            "systems": {
                name: {
                    "all": {"recall@5": round(recall(per_query, name), 3), "mrr": round(mrr(per_query, name), 3)},
                    "hard": {"recall@5": round(recall(hard, name), 3), "mrr": round(mrr(hard, name), 3)},
                    "mrr_by_category": {c: round(mrr(by_cat[c], name), 3) for c in categories},
                }
                for name in systems
            },
            "per_query": [
                {
                    "query": r["query"],
                    "category": q.get("category", "basic"),
                    "first_relevant_rank": {name: r[name][1] for name in systems},
                }
                for r, q in zip(per_query, queries, strict=True)
            ],
        }
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nsaved → {args.save}")


if __name__ == "__main__":
    main()
