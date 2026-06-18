# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Ultra-mode writer: runs System1 like pro, then fire-and-forget enqueues the
(user, app) pair for later System2 distillation drained by run_system2_pending.
"""
from dual_mem.registry import ComponentFactory
from dual_mem.system2.system2_agent import System2Agent
from dual_mem.writer.memory_writer import MemoryWriter, WriteResult


class System2Writer:
    """Ultra writer wrapping the System1 writer and queuing async System2 distillation."""

    def __init__(self, *, factory: ComponentFactory, agent_mode: str):
        self.factory = factory
        self.inner = MemoryWriter(factory=factory, agent_mode=agent_mode)
        self.processed_pairs: set[tuple[str, str]] = set()

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
        """Run the System1 write, enqueue the (user, app) pair for System2, and return."""
        result = await self.inner.write(
            content=content,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            request_id=request_id,
            memory_at=memory_at,
        )
        self.factory.cache.enqueue_s2_task(user_id, app_id)
        return result

    async def run_system2_pending(self) -> int:
        """Drain the System2 queue, running the agent per task; return tasks processed."""
        agent = System2Agent(factory=self.factory)
        self.processed_pairs = set()
        processed = 0
        while True:
            task = self.factory.cache.dequeue_s2_task()
            if task is None:
                break
            agent.run(app_id=task["app_id"], user_id=task["user_id"])
            self.processed_pairs.add((task["app_id"], task["user_id"]))
            processed += 1
        return processed
