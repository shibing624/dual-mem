# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Synchronous-write path: persist a raw L1 memory node, log it, and run the
System1 cognition pipeline (extract -> commit decision -> fast-write). In dual mode, L2/L4
writes queue reconcile work for explicit digest(). L7 intentions and optional L3 summaries
are written inline. dual-mem requires both LLM and embedding providers.
"""
import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.registry import ComponentFactory
from dual_mem.sdk_models import ChatMessage
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
        messages: list[ChatMessage] | None = None,
    ) -> WriterOutcome:
        """Persist content according to the configured memory mode."""
        settings = self.factory.settings
        is_system1 = settings.mode == "system1"
        iso_key = _content_iso_key(
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            scope=settings.content_hash_scope,
        )
        content_hash: str | None = None
        if settings.content_hash_dedup and not is_system1:
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

        if is_system1 or not settings.skip_l1_vector_when_derived:
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

        # reconcile_sync is a dual-pipeline feature: honor it even in system1
        # mode by delegating to the full dual agent pipeline.
        if is_system1 and not settings.reconcile_sync:
            agent = MemAgent(factory=self.factory)
            extra_node_ids, commit_result, is_ephemeral = await agent.run_system1(
                content=content,
                raw_node=node,
                messages=messages,
                request_id=request_id,
            )
            if extra_node_ids:
                # Derived memories carry their own embeddings now, so the raw
                # chunk leaves the recall pool (shadow), matching dual mode.
                node.status = MemoryStatus.SHADOW
                await asyncio.to_thread(self.factory.vector.upsert, [node])
            return WriterOutcome(
                memory_id=node.node_id,
                extra_node_ids=extra_node_ids,
                commit_passed=commit_result.passed,
                is_ephemeral=is_ephemeral,
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
            messages=messages,
            source_node_id=node.node_id,
        )

        if extra_node_ids:
            node.status = MemoryStatus.SHADOW
            if not settings.skip_l1_vector_when_derived:
                await asyncio.to_thread(self.factory.vector.upsert, [node])
        elif settings.skip_l1_vector_when_derived:
            node.embedding = await self.factory.embed.embed_queued(content)
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

    async def write_raw(
        self,
        *,
        content: str,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        memory_at: int | None = None,
    ) -> WriterOutcome:
        """Persist L1_RAW only — no extract / reconcile."""
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
        return WriterOutcome(memory_id=node.node_id, is_ephemeral=True)

    async def distill(
        self,
        *,
        content: str,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        request_id: str,
        source_node_ids: list[str],
        memory_at: int | None = None,
        messages: list[ChatMessage] | None = None,
    ) -> WriterOutcome:
        """Extract from existing L1 nodes; do not write another L1."""
        raw = (
            self.factory.vector.get(source_node_ids[0]) if source_node_ids else None
        )
        if raw is None:
            return WriterOutcome(memory_id="", commit_passed=False, is_ephemeral=True)
        settings = self.factory.settings
        is_system1 = settings.mode == "system1"
        agent = MemAgent(factory=self.factory)
        if is_system1 and not settings.reconcile_sync:
            extra_node_ids, commit_result, is_ephemeral = await agent.run_system1(
                content=content,
                raw_node=raw,
                messages=messages,
                request_id=request_id,
            )
        else:
            extra_node_ids, commit_result, is_ephemeral = await agent.run(
                content=content,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                request_id=request_id,
                memory_at=memory_at if memory_at is not None else raw.memory_at,
                messages=messages,
                source_node_id=raw.node_id,
            )
        if extra_node_ids:
            for nid in source_node_ids:
                node = self.factory.vector.get(nid)
                if node is None:
                    continue
                node.status = MemoryStatus.SHADOW
                await asyncio.to_thread(self.factory.vector.upsert, [node])
        return WriterOutcome(
            memory_id=raw.node_id,
            extra_node_ids=extra_node_ids,
            commit_passed=commit_result.passed,
            is_ephemeral=is_ephemeral,
        )


__all__ = ["MemoryWriter", "WriterOutcome"]
