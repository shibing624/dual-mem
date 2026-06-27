# -*- coding: utf-8 -*-
"""Profile reverse path: VDB hits → supporting L6 schemas via graph DERIVED_FROM edges."""
import asyncio
import logging

from dual_mem.types import Layer

logger = logging.getLogger("dual_mem.retrieval.profile_evidence")

_L6_LAYER = Layer.L6_SCHEMA.value


async def reverse_lookup_l6(
    graph_store,
    vdb_node_ids: list[str],
    *,
    limit: int = 50,
) -> list[dict]:
    """Reverse-lookup L6 schemas that cite the given VDB node ids; rank by support count."""
    if graph_store is None or not vdb_node_ids:
        return []

    try:
        rows = await asyncio.to_thread(
            graph_store.find_referencing_memories, vdb_node_ids, limit=limit
        )
    except Exception as exc:
        logger.debug("[profile-evidence] reverse_lookup_l6 failed: %s", exc)
        return []

    grouped: dict[str, dict] = {}
    for row in rows or []:
        if row.get("layer") != _L6_LAYER:
            continue
        nid = row.get("node_id")
        if not nid:
            continue
        entry = grouped.get(nid)
        if entry is None:
            entry = {
                "node_id": nid,
                "content": row.get("content", ""),
                "confidence": row.get("confidence"),
                "_evidence_ids": set(),
            }
            grouped[nid] = entry
        ev = row.get("evidence_vdb_id")
        if ev:
            entry["_evidence_ids"].add(ev)

    if not grouped:
        return []

    max_support = max(len(g["_evidence_ids"]) or 1 for g in grouped.values())
    hits: list[dict] = []
    for g in grouped.values():
        support = len(g["_evidence_ids"]) or 1
        hits.append(
            {
                "node_id": g["node_id"],
                "content": g["content"],
                "layer": _L6_LAYER,
                "score": support / max_support,
                "source": "profile_reverse",
                "confidence": g.get("confidence"),
                "_support_count": support,
                "node": None,
            }
        )
    hits.sort(key=lambda h: h["_support_count"], reverse=True)
    return hits
