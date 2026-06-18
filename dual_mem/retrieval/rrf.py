"""Reciprocal Rank Fusion —— 多路召回的 rank 级融合（忠实复现源码）。

公式：score(d) = Σ_c [ w_c / (k + rank_c(d)) ]。只看 rank 不看绝对分数，
规避不同通道分数尺度不可比；出现在多路的 doc 天然累加。
"""

from typing import Any

RRF_K = 60


def rrf_fuse(
    channels: dict[str, list[dict[str, Any]]],
    weights: dict[str, float] | None = None,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """RRF 融合多路召回。

    每路 hits 必须按各自分数降序排好，且每条含 ``node_id``。返回按 rrf_score
    降序的合并列表，每条含 ``node_id`` / ``node`` / ``rrf_score`` /
    ``rrf_rank_by_channel`` / ``per_channel_score``。
    """
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
