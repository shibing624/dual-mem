# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Synchronous-write path: persist a raw L1 memory node, log it, and run the
System1 cognition pipeline (gate -> extract -> fast-write) which queues async reconcile work
and writes any L7 intentions / L3 summary in line. dual-mem requires LLM + embedding API
keys; there is no embedding-only / no-LLM mode.
"""
import hashlib
import logging
from dataclasses import dataclass, field

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.registry import ComponentFactory
from dual_mem.sdk_models import GateResult
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.writer")


def _content_iso_key(*, app_id: str, user_id: str, agent_id: str) -> str:
    return f"{app_id}::{user_id}::{agent_id}"


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
        settings = self.factory.settings
        iso_key = _content_iso_key(app_id=app_id, user_id=user_id, agent_id=agent_id)
        content_hash: str | None = None
        if settings.content_hash_dedup:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            cached = self.factory.cache.get_content_hash_outcome(iso_key, content_hash)
            if cached is not None:
                logger.debug(
                    "content_hash hit user=%s hash=%s memory_id=%s",
                    user_id,
                    content_hash[:8],
                    cached.get("memory_id"),
                )
                return WriterOutcome(
                    memory_id=str(cached["memory_id"]),
                    extra_node_ids=list(cached.get("extra_node_ids") or []),
                    gate_passed=bool(cached.get("gate_passed", True)),
                    gate_score=cached.get("gate_score"),
                    is_ephemeral=bool(cached.get("is_ephemeral", False)),
                )

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

        gate_turn_embeddings: list[list[float]] | None = None
        if user_queries and settings.embed_merge_l1_gate:
            # One embed RTT for L1 + Gate user turns; bypasses embed_queued coalescing — enable
            # only when single-write latency matters more than concurrent write throughput.
            batch_texts = [content, *user_queries]
            vectors = await self.factory.embed.embed_batch(batch_texts)
            embedding = vectors[0]
            gate_turn_embeddings = vectors[1:]
        else:
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
            gate_turn_embeddings=gate_turn_embeddings,
        )

        if extra_node_ids:
            node.status = MemoryStatus.SHADOW
            self.factory.vector.upsert([node])

        outcome = WriterOutcome(
            memory_id=node.node_id,
            extra_node_ids=extra_node_ids,
            gate_passed=gate_result.passed,
            gate_score=gate_result.gate_score,
            is_ephemeral=is_ephemeral,
        )
        if settings.content_hash_dedup and content_hash is not None:
            self.factory.cache.set_content_hash_outcome(
                iso_key,
                content_hash,
                {
                    "memory_id": outcome.memory_id,
                    "extra_node_ids": outcome.extra_node_ids,
                    "gate_passed": outcome.gate_passed,
                    "gate_score": outcome.gate_score,
                    "is_ephemeral": outcome.is_ephemeral,
                },
            )
        return outcome


# Re-export GateResult for callers that want to inspect gate decisions on a write.
__all__ = ["MemoryWriter", "WriterOutcome", "GateResult"]
