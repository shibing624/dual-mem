# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async reader backed by the single hybrid retrieval pipeline.
"""
import time

from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.hybrid_engine import search_hybrid
from dual_mem.sdk_models import MemoryItem, ReadResult, SearchMemories
from dual_mem.types import MemoryNode

class Reader:
    """Embed a query, run hybrid recall, and return grouped memory items."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    async def search(
        self,
        *,
        query: str,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None = None,
        session_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        include_derived: bool = True,
        created_after: int | None = None,
        request_id: str | None = None,
        collect_trace: bool = False,
    ) -> tuple[SearchMemories, ReadResult | None]:
        """Recall memories using explicit filters; no query-intent guessing is performed."""
        start = time.perf_counter()
        rid = request_id or "search"
        query_embedding = await self.factory.embed.embed_queued(query)
        memories = await search_hybrid(
            factory=self.factory,
            query=query,
            query_embedding=query_embedding,
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            limit=limit,
            min_score=min_score,
            profile_limit=profile_limit,
            profile_min_score=profile_min_score,
            intention_limit=intention_limit,
            include_derived=include_derived,
            created_after=created_after,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        trace = None
        if collect_trace:
            trace = ReadResult(
                memories=memories,
                final_count=(
                    len(memories.profile)
                    + len(memories.proactive)
                    + len(memories.normal)
                ),
                elapsed_ms=elapsed_ms,
            )
        self.factory.cache.log_pipeline(
            request_id=rid,
            stage="READ_HYBRID",
            payload={
                "profile": len(memories.profile),
                "normal": len(memories.normal),
                "proactive": len(memories.proactive),
                "include_derived": include_derived,
            },
        )
        return memories, trace

    @staticmethod
    def memory_node_to_item(node: MemoryNode) -> MemoryItem:
        """Convert a stored node used by get/list into the public memory model."""
        return MemoryItem(
            memory_id=node.node_id,
            content=node.content,
            category=node.category.value,
            score=node.score,
            tags=list(node.tags),
            session_id=node.session_id,
            memory_at=node.memory_at,
            gmt_created=node.gmt_created,
            gmt_modified=node.gmt_modified,
        )
