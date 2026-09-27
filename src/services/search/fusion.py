"""Reciprocal Rank Fusion: merge ranked lists whose scores are on different scales (BM25 vs cosine).

score(doc) = Σ over lists  1 / (k + rank_in_that_list)      (rank starts at 1; missing from a list → adds 0)

Only positions matter, so no score normalisation or weights to tune; documents ranked well by BOTH
retrievers rise to the top. k = 60 is the value from the original paper (Cormack et al., 2009).
"""

from collections.abc import Hashable


def reciprocal_rank_fusion(rankings: dict[str, list[Hashable]], k: int = 60) -> dict[Hashable, dict]:
    """rankings: {"keyword": [id1, id2, …], "vector": [id3, id1, …]} → {id: {"rrf": score, "ranks": {...}}}"""
    fused: dict[Hashable, dict] = {}
    for name, ids in rankings.items():
        for rank, doc_id in enumerate(ids, start=1):
            entry = fused.setdefault(doc_id, {"rrf": 0.0, "ranks": {}})
            entry["rrf"] += 1.0 / (k + rank)
            entry["ranks"][name] = rank
    return dict(sorted(fused.items(), key=lambda item: item[1]["rrf"], reverse=True))
