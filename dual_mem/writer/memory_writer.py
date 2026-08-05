# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Synchronous-write path: persist a raw L1 memory node, log it, and run the
System1 cognition pipeline (extract -> commit decision -> fast-write). In dual mode, L2/L4
writes queue reconcile work for explicit digest(). L7 intentions and optional L3 summaries
are written inline. dual-mem requires both LLM and embedding providers.
"""
import hashlib
import logging
import asyncio
from dataclasses import dataclass, field

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.writer")


def _content_iso_key(
    *,
    app_id: str,
    user_id: str,
    agent_id: str,
    session_id: str,
    scope: str = "session",
) -> str:
    """Dedup key for content_hash. ``session`` scope keys per app/user/agent/session (strict);
    ``user`` scope keys per app/user (cross-session/agent dedup, higher hit rate)."""
    if scope == "user":
        return f"{app_id}::{user_id}"
    return f"{app_id}::{user_id}::{agent_id}::{session_id}"


@dataclass
class WriterOutcome:
    """Internal write-path outcome before the public ``sdk_models.WriteResult`` wrapper."""

    memory_id: str
    extra_node_ids: list[str] = field(default_factory=list)
    commit_passed: bool = True
    is_ephemeral: bool = False
    deduplicated: bool = False


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
    ) -> WriterOutcome:
        """Persist content as a raw node, then derive and link extra memories via System1."""
        settings = self.factory.settings
        iso_key = _content_iso_key(
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            scope=settings.content_hash_scope,
        )
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
                    commit_passed=bool(cached.get("commit_passed", True)),
                    is_ephemeral=bool(cached.get("is_ephemeral", False)),
                    deduplicated=True,
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

        node.embedding = await self.factory.embed.embed_queued(content)
        await asyncio.to_thread(self.factory.vector.upsert, [node])
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
        extra_node_ids, commit_result, is_ephemeral = await agent.run(
            content=content,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            request_id=request_id,
            memory_at=memory_at,
        )

        if extra_node_ids:
            node.status = MemoryStatus.SHADOW
            await asyncio.to_thread(self.factory.vector.upsert, [node])

        outcome = WriterOutcome(
            memory_id=node.node_id,
            extra_node_ids=extra_node_ids,
            commit_passed=commit_result.passed,
            is_ephemeral=is_ephemeral,
        )
        if settings.content_hash_dedup and content_hash is not None:
            self.factory.cache.set_content_hash_outcome(
                iso_key,
                content_hash,
                {
                    "memory_id": outcome.memory_id,
                    "extra_node_ids": outcome.extra_node_ids,
                    "commit_passed": outcome.commit_passed,
                    "is_ephemeral": outcome.is_ephemeral,
                },
            )
        return outcome


__all__ = ["MemoryWriter", "WriterOutcome"]
