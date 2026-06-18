# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Synchronous write path (lite/pro): persists a raw memory node, logs it,
and in full agent mode runs System1 cognition to derive extra memories.
"""
from dataclasses import dataclass, field

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryNode, MemoryStatus


@dataclass
class WriteResult:
    """Outcome of a write: the raw memory id plus any cognition-derived node ids."""

    memory_id: str
    extra_node_ids: list[str] = field(default_factory=list)


class MemoryWriter:
    """Lite/pro writer: persist the raw memory and optionally run System1 cognition."""

    def __init__(self, *, factory: ComponentFactory, agent_mode: str):
        self.factory = factory
        self.agent_mode = agent_mode

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
    ) -> WriteResult:
        """Persist content as a raw node and, in full mode, derive and link extra memories."""
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
        node.embedding = self.factory.embed.embed(content)
        self.factory.vector.upsert([node])
        self.factory.history.append(
            event="ADD",
            node_id=node.node_id,
            user_id=user_id,
            old=None,
            new=node.to_metadata(),
        )

        extra_node_ids: list[str] = []
        if self.agent_mode == "full":
            extra_node_ids = await self._run_cognition(
                raw=node,
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
                self.factory.vector.upsert([node])

        return WriteResult(memory_id=node.node_id, extra_node_ids=extra_node_ids)

    async def _run_cognition(
        self,
        *,
        raw: MemoryNode,
        content: str,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        request_id: str,
        memory_at: int | None,
    ) -> list[str]:
        """Run the System1 MemAgent over the raw node and return derived node ids."""
        agent = MemAgent(factory=self.factory)
        return agent.run(
            raw_node=raw,
            content=content,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            request_id=request_id,
            memory_at=memory_at,
        )
