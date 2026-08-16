# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Dual-mode writer with explicit System2 digestion. Writes run System1 and
record a pending cognition scope; reconcile and System2 inference run only via digest().
"""
import logging
import time

from dual_mem.locks import LockRegistry
from dual_mem.registry import ComponentFactory
from dual_mem.system2.reconciler_worker import (
    ReconcilerWorker,
    link_evolution_chains_heuristic,
)
from dual_mem.system2.system2_agent import System2Agent
from dual_mem.types import Layer
from dual_mem.writer.memory_writer import MemoryWriter, WriterOutcome

logger = logging.getLogger("dual_mem.system2.writer")


class System2Writer:
    """Run System1 on write and defer all System2 work to explicit ``digest()`` calls."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory
        self.inner = MemoryWriter(factory=factory)
        self._user_locks = LockRegistry()
        self.last_digest_stats: dict[str, float] = {}

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
        """Run System1 and record one deduplicated cognition task for later digestion."""
        result = await self.inner.write(
            content=content,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            request_id=request_id,
            memory_at=memory_at,
        )
        has_new_facts = False
        for node_id in result.extra_node_ids:
            node = self.factory.vector.get(node_id)
            if node is not None and node.layer in (Layer.L2_FACT, Layer.L4_IDENTITY):
                has_new_facts = True
                break
        if not result.deduplicated and has_new_facts:
            self.factory.cache.enqueue_s2_task(
                user_id=user_id,
                app_id=app_id,
                agent_id=agent_id,
            )
        return result

    async def digest_pending(self) -> int:
        """Drain pending cognition scopes and return the number of scopes processed."""
        self.last_digest_stats = {
            "reconcile_sec": 0.0,
            "s2_agent_sec": 0.0,
            "reconcile_tasks": 0.0,
            "s2_agent_runs": 0.0,
        }
        scopes: set[tuple[str, str, str]] = {
            (
                row["app_id"],
                row["user_id"],
                row.get("agent_id") or "",
            )
            for row in self.factory.cache.list_pending_reconcile_scopes()
        }
        for task in self.factory.cache.list_pending_s2_scopes():
            scopes.add(
                (
                    task["app_id"],
                    task["user_id"],
                    task.get("agent_id") or "",
                )
            )

        for app_id, user_id, agent_id in sorted(scopes):
            await self._digest_user(
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
            )
            self.factory.cache.mark_s2_scope_done(
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
            )
        return len(scopes)

    async def _digest_user(self, *, app_id: str, user_id: str, agent_id: str) -> None:
        """Serialize and run reconcile followed by System2 inference for one scope."""
        key = f"{app_id}::{user_id}::{agent_id}"
        async with self._user_locks.get(key):
            settings = self.factory.settings
            t0 = time.perf_counter()
            if settings.reconcile_skip_llm:
                tasks = self.factory.cache.list_pending_reconcile_tasks(
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                )
                # NOTE(dual_vs_hy): link_evolution_chains_heuristic 被删除 —— 它按 tag 分组
                # 批量制造 supersedes/superseded_by 演化链指针（实测 62% 记忆被打链），
                # 检索时 expand_evolution_chains 把这些链全量注入 QA 上下文，挤占事实证据。
                # 对标 hy-memory ultra：hy 的 S2 只用 single-shot JSON ops 建 L6 图，不建
                # heuristic chain。原始 fact 保持 ACTIVE 不隐藏，队列只排空不建链。
                for task in tasks:
                    self.factory.cache.mark_reconcile_task_done(task["id"])
                reconcile_tasks = len(tasks)
            else:
                worker = ReconcilerWorker(factory=self.factory)
                reconcile_tasks = await worker.reconcile_pending(
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                )
            reconcile_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            agent = System2Agent(factory=self.factory)
            await agent.run(app_id=app_id, user_id=user_id, agent_id=agent_id)
            agent_sec = time.perf_counter() - t1

            self.last_digest_stats["reconcile_sec"] += reconcile_sec
            self.last_digest_stats["s2_agent_sec"] += agent_sec
            self.last_digest_stats["reconcile_tasks"] += reconcile_tasks
            self.last_digest_stats["s2_agent_runs"] += 1
            logger.info(
                "[s2] digest user=%s reconcile=%.1fs(tasks=%d) s2_agent=%.1fs",
                user_id,
                reconcile_sec,
                reconcile_tasks,
                agent_sec,
            )


__all__ = ["System2Writer"]
