# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Synchronous-write path: persist a raw L1 memory node, log it, and run the
System1 cognition pipeline (gate -> extract -> fast-write) which queues async reconcile work
and writes any L7 intentions / L3 summary in line. dual-mem requires LLM + embedding API
keys; there is no embedding-only / no-LLM mode.
"""
import logging
from dataclasses import dataclass, field

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.registry import ComponentFactory
from dual_mem.sdk_models import GateResult
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.writer")


@dataclass
class WriterOutcome:
    """Internal write-path outcome before the public ``sdk_models.WriteResult`` wrapper."""

    memory_id: str
    extra_node_ids: list[str] = field(default_factory=list)
    gate_passed: bool = True
    gate_score: float | None = None
    is_ephemeral: bool = False


class MemoryWriter:
    """system1 writer: persist the raw memory and run the System1 cognition pipeline."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    async def write(
        self,
        *,
        content: str,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        request_id: str,
        memory_at: int | None = None,
        user_queries: list[str] | None = None,
        agent_context: str | None = None,
    ) -> WriterOutcome:
        """Persist content as a raw node, then derive and link extra memories via System1.

        ``user_queries`` carries the per-turn user texts for multi-turn writes; the gate uses
        them for novelty=max-across-turns. None means single-turn (fall back to ``content``).
        """
        node = MemoryNode(
            content=content,
            layer=Layer.L1_RAW,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            status=MemoryStatus.ACTIVE,
            memory_at=memory_at,
        )
        embedding = await self.factory.embed.embed_queued(content)
        node.embedding = embedding
        self.factory.vector.upsert([node])
        self.factory.history.append(
            event="ADD",
            node_id=node.node_id,
            user_id=user_id,
            old=None,
            new=node.to_metadata(),
        )
        logger.debug(
            "write_raw user=%s memory_id=%s len=%d",
            user_id, node.node_id, len(content),
        )

        agent = MemAgent(factory=self.factory)
        extra_node_ids, gate_result, is_ephemeral = await agent.run(
            raw_node=node,
            content=content,
            embedding=embedding,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            request_id=request_id,
            memory_at=memory_at,
            user_queries=user_queries,
            agent_context=agent_context,
        )

        if extra_node_ids:
            node.status = MemoryStatus.SHADOW
            self.factory.vector.upsert([node])

        return WriterOutcome(
            memory_id=node.node_id,
            extra_node_ids=extra_node_ids,
            gate_passed=gate_result.passed,
            gate_score=gate_result.gate_score,
            is_ephemeral=is_ephemeral,
        )


# Re-export GateResult for callers that want to inspect gate decisions on a write.
__all__ = ["MemoryWriter", "WriterOutcome", "GateResult"]
