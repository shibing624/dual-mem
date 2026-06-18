# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Reciprocal Rank Fusion for combining multi-channel recall by rank rather
than absolute score: score(d) = Σ_c w_c / (k + rank_c(d)).
"""
from typing import Any

RRF_K = 60


def rrf_fuse(
    channels: dict[str, list[dict[str, Any]]],
    weights: dict[str, float] | None = None,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse pre-ranked recall channels into one list sorted by combined RRF score."""
    weights = weights or {}
    acc: dict[str, dict[str, Any]] = {}

    for channel_name, hits in channels.items():
        w = float(weights.get(channel_name, 1.0))
        if w == 0 or not hits:
            continue
        for rank, hit in enumerate(hits, start=1):
            nid = hit.get("node_id")
            if not nid:
                continue
            contrib = w / (k + rank)
            entry = acc.get(nid)
            if entry is None:
                entry = {
                    "node_id": nid,
                    "node": hit.get("node"),
                    "rrf_score": 0.0,
                    "rrf_rank_by_channel": {},
                    "per_channel_score": {},
                }
                acc[nid] = entry
            elif entry["node"] is None and hit.get("node") is not None:
                entry["node"] = hit.get("node")
            entry["rrf_score"] += contrib
            entry["rrf_rank_by_channel"][channel_name] = rank
            entry["per_channel_score"][channel_name] = float(hit.get("score", 0.0))

    fused = list(acc.values())
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused
