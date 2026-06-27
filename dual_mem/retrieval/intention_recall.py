# -*- coding: utf-8 -*-
"""L7 intention recall from VDB (proactive route)."""
import asyncio
import logging

from dual_mem.isolation import build_filter
from dual_mem.types import Layer, MemoryStatus

logger = logging.getLogger("dual_mem.retrieval.intention_recall")


async def recall_intentions(
    vector,
    query_embedding: list[float],
    *,
    app_ids: list[str],
    user_id: str,
    agent_ids: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Recall ACTIVE L7_INTENTION nodes from the vector store."""
    if limit <= 0:
        return []
    where = build_filter(
        app_ids=app_ids,
        user_id=user_id,
        agent_ids=agent_ids,
        layers=[Layer.L7_INTENTION],
        statuses=[MemoryStatus.ACTIVE],
    )
    try:
        nodes = await asyncio.to_thread(
            vector.query, embedding=query_embedding, where=where, top_k=limit
        )
    except Exception as exc:
        logger.debug("[intention] recall failed: %s", exc)
        return []

    return [
        {
            "node_id": n.node_id,
            "score": n.score,
            "node": n,
            "layer": Layer.L7_INTENTION.value,
            "source": "vdb_intention",
        }
        for n in nodes
    ]
