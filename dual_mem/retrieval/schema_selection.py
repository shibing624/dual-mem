# -*- coding: utf-8 -*-
"""Choose the System 2 schemas that accompany a search result.

Schemas never compete with System 1 memories for result slots. They are selected
*after* the System 1 result is final, from two routes:

- **reverse**: schemas whose DERIVED_FROM evidence includes a retrieved memory,
  ranked by how many retrieved memories they explain. A schema reached this way is
  anchored in the evidence the reader already sees.
- **forward**: schemas whose retrieval text (situation + behaviour) is close to the
  query, kept only above ``forward_min_score`` because nothing in the result vouches
  for them.

The two rankings are fused by reciprocal rank (their scores are not comparable),
and each selected schema reports which retrieved memories support it, so the
reader can see the pattern next to its concrete evidence.
"""
from __future__ import annotations

from collections.abc import Callable

from dual_mem.retrieval import rrf
from dual_mem.types import Layer, MemoryNode


def select_schemas(
    *,
    vector,
    graph,
    evidence: list[MemoryNode],
    schema_hits: list[dict],
    scope_filter: Callable[[MemoryNode], bool],
    limit: int,
    forward_min_score: float,
    min_support: int = 1,
    schema_kinds: tuple[str, ...] = ("schema", "core_schema"),
) -> list[dict]:
    if limit <= 0:
        return []
    evidence_ids = [n.node_id for n in evidence]
    supports: dict[str, set[str]] = {}
    if graph is not None and evidence_ids:
        for row in graph.find_referencing_memories(evidence_ids, limit=max(limit * 10, 50)):
            if row.get("layer") == Layer.L3_DERIVED.value and row.get("evidence_vdb_id"):
                supports.setdefault(row["node_id"], set()).add(row["evidence_vdb_id"])
    reverse = sorted(
        (nid for nid, ev in supports.items() if len(ev) >= min_support),
        key=lambda nid: -len(supports[nid]),
    )
    forward = [h["node_id"] for h in schema_hits if h["score"] >= forward_min_score]
    forward_score = {h["node_id"]: h["score"] for h in schema_hits}

    fused = rrf.rrf_fuse({
        "reverse": [{"node_id": nid} for nid in reverse],
        "forward": [{"node_id": nid} for nid in forward],
    })
    heads = vector.get_by_ids([row["node_id"] for row in fused])
    out: list[dict] = []
    top = max((row["rrf_score"] for row in fused), default=0.0) or 1.0
    for row in fused:
        node = heads.get(row["node_id"])
        if (node is None or node.layer != Layer.L3_DERIVED or node.kind not in schema_kinds
                or not scope_filter(node)):
            continue
        support = sorted(supports.get(node.node_id, set()))
        node.custom = {**(node.custom or {}), "supporting_result_ids": support}
        out.append({
            "node_id": node.node_id,
            "node": node,
            # Order within the schema group only; never compared with fact scores.
            "score": row["rrf_score"] / top,
            "source": "schema",
            "_forward_score": forward_score.get(node.node_id),
        })
        if len(out) >= limit:
            break
    return out


__all__ = ["select_schemas"]
