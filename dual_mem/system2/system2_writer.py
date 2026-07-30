# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: dual-mode writer that wraps the System1 writer and dispatches asynchronous
System2 cognition. Supports three trigger modes (per_write / manual / scheduled) with a
per-user lock so concurrent writes for the same user serialize through reconcile -> S2
agent -> cross-domain sweeper, while different users run in parallel. Reconsolidation
tasks (enqueued by the read-side hook) refresh recall timestamps without extra LLM calls.
"""
import asyncio
import logging
import time

from dual_mem.locks import LockRegistry
from dual_mem.registry import ComponentFactory
from dual_mem.system2.cross_domain_sweeper import CrossDomainSweeper
from dual_mem.system2.reconciler_worker import (
    ReconcilerWorker,
    link_evolution_chains_heuristic,
)
from dual_mem.system2.system2_agent import System2Agent
from dual_mem.writer.memory_writer import MemoryWriter, WriterOutcome

logger = logging.getLogger("dual_mem.system2.writer")

_PER_WRITE = "per_write"
_SCHEDULED = "scheduled"


class System2Writer:
    """dual writer: synchronous System1 + asynchronous System2 (reconcile/agent/sweeper)."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory
        self.inner = MemoryWriter(factory=factory)
        self.processed_pairs: set[tuple[str, str]] = set()
        # LRU-bounded per-(app,user,agent) drain locks so a long-lived server does not
        # accumulate one Lock per identity forever.
        self._user_locks = LockRegistry()
        self._scheduled_task: asyncio.Task | None = None
        self._scheduled_started: bool = False
        # Per-digest timing breakdown (reconcile vs S2 agent vs sweeper). Reset at the
        # start of each _digest_pending drain; surfaced via MemoryClient.digest().
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
        """Run System1, queue an S2 cognition task, and dispatch per the trigger mode."""
        result = await self.inner.write(
            content=content,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            request_id=request_id,
            memory_at=memory_at,
        )

        # Hy parity: manual mode does not enqueue S2 on write (explicit digest() only).
        # per_write / scheduled keep the sqlite queue for background/scheduled drains.
        mode = self.factory.settings.system2_trigger_mode
        if mode in (_PER_WRITE, _SCHEDULED):
            self.factory.cache.enqueue_s2_task(
                user_id=user_id, app_id=app_id, agent_id=agent_id
            )

        if mode == _PER_WRITE:
            task = asyncio.create_task(
                self._digest_user_safely(app_id=app_id, user_id=user_id, agent_id=agent_id)
            )
            task.add_done_callback(_log_task_exception)
            logger.debug("[s2] background drain dispatched user=%s", user_id)
        elif mode == _SCHEDULED:
            self._ensure_scheduled_loop()
        # _MANUAL: nothing to do; user must call client.digest()

        return result

    async def _digest_reconsolidation_pending(self) -> int:
        """Drain any pending ``task_type=reconsolidation`` tasks. Safe to call from the
        read path (per_write) so search-triggered reconsolidation does not wait for the
        next write. Returns the count of reconsolidation tasks processed."""
        cache = self.factory.cache
        processed = 0
        while True:
            task = cache.dequeue_s2_task(task_type="reconsolidation")
            if task is None:
                break
            await self._run_reconsolidation_safely(task)
            processed += 1
        return processed

    async def _digest_pending(self) -> int:
        """Drain every pending S2 task across all users; return tasks processed."""
        cache = self.factory.cache
        self.processed_pairs = set()
        self.last_digest_stats = {
            "reconcile_sec": 0.0,
            "s2_agent_sec": 0.0,
            "sweeper_sec": 0.0,
            "reconcile_tasks": 0.0,
            "s2_agent_runs": 0.0,
            "reconsolidation_tasks": 0.0,
        }
        processed = 0
        drained_pairs: set[tuple[str, str, str]] = set()

        # Manual Hy parity: reconcile pending scopes even without s2 queue entries.
        for scope in cache.list_pending_reconcile_scopes():
            key = (
                scope["app_id"],
                scope["user_id"],
                scope.get("agent_id") or "",
            )
            if key in drained_pairs:
                continue
            drained_pairs.add(key)
            await self._digest_user_safely(
                app_id=key[0], user_id=key[1], agent_id=key[2]
            )
            self.processed_pairs.add((key[0], key[1]))
            processed += 1

        while True:
            task = cache.dequeue_s2_task()
            if task is None:
                break
            if task.get("task_type", "cognition") == "reconsolidation":
                await self._run_reconsolidation_safely(task)
                self.last_digest_stats["reconsolidation_tasks"] += 1
                processed += 1
                continue
            key = (
                task["app_id"],
                task["user_id"],
                task.get("agent_id") or "",
            )
            if key not in drained_pairs:
                drained_pairs.add(key)
                await self._digest_user_safely(
                    app_id=key[0], user_id=key[1], agent_id=key[2]
                )
                processed += 1
            self.processed_pairs.add((task["app_id"], task["user_id"]))
        return processed

    async def aclose(self) -> None:
        """Cancel any background scheduled loop; safe to call multiple times."""
        if self._scheduled_task is not None and not self._scheduled_task.done():
            self._scheduled_task.cancel()
            try:
                await self._scheduled_task
            except (asyncio.CancelledError, Exception):
                pass
        self._scheduled_task = None
        self._scheduled_started = False

    # ---- Internal helpers ------------------------------------------------------------

    async def _digest_user_safely(
        self, *, app_id: str, user_id: str, agent_id: str = ""
    ) -> None:
        """Per-user serialized drain: reconcile -> S2 agent -> optional sweeper."""
        key = f"{app_id}::{user_id}::{agent_id}"
        lock = self._user_locks.get(key)
        async with lock:
            try:
                await self._digest_user(app_id=app_id, user_id=user_id, agent_id=agent_id)
            except Exception as exc:
                logger.exception("[s2] drain failed for %s: %s", key, exc)

    async def _digest_user(self, *, app_id: str, user_id: str, agent_id: str) -> None:
        """Run reconcile + S2 agent + (optional) cross-domain sweeper for one user."""
        settings = self.factory.settings
        t_reconcile = 0.0
        n_reconcile = 0

        if settings.reconcile_skip_llm:
            # non_destructive fast-path: L2 facts already ACTIVE from Extractor. Reconcile
            # LLM output would be stripped anyway (no supersede/DELETE), so skip it entirely
            # and just drain the queue. Saves ~47 LLM calls (~127s) per question.
            cache = self.factory.cache
            drained = 0
            while cache.dequeue_reconcile_task(
                app_id=app_id, user_id=user_id, agent_id=agent_id
            ) is not None:
                drained += 1
            n_reconcile = drained
            logger.info("[s2] reconcile LLM skipped (skip_llm): drained %d tasks", drained)
            if settings.reconcile_link_chains_heuristic:
                t0 = time.perf_counter()
                n_links = link_evolution_chains_heuristic(
                    self.factory, app_id=app_id, user_id=user_id, agent_id=agent_id
                )
                t_reconcile = time.perf_counter() - t0
                logger.info(
                    "[s2] heuristic chain-linking: wired %d evolution edges", n_links
                )
        else:
            worker = ReconcilerWorker(factory=self.factory)
            t0 = time.perf_counter()
            n_reconcile = await worker.reconcile_pending(
                app_id=app_id, user_id=user_id, agent_id=agent_id
            )
            t_reconcile = time.perf_counter() - t0

        agent = System2Agent(factory=self.factory)
        t1 = time.perf_counter()
        await agent.run(app_id=app_id, user_id=user_id, agent_id=agent_id)
        t_agent = time.perf_counter() - t1

        t_sweep = 0.0
        if self.factory.settings.cross_domain_enable:
            sweeper = CrossDomainSweeper(factory=self.factory)
            t2 = time.perf_counter()
            await sweeper.run(app_id=app_id, user_id=user_id, agent_id=agent_id)
            t_sweep = time.perf_counter() - t2

        self.last_digest_stats["reconcile_sec"] += t_reconcile
        self.last_digest_stats["s2_agent_sec"] += t_agent
        self.last_digest_stats["sweeper_sec"] += t_sweep
        self.last_digest_stats["reconcile_tasks"] += n_reconcile
        self.last_digest_stats["s2_agent_runs"] += 1
        logger.info(
            "[s2] digest user=%s reconcile=%.1fs(tasks=%d) s2_agent=%.1fs sweeper=%.1fs",
            user_id, t_reconcile, n_reconcile, t_agent, t_sweep,
        )

    async def _run_reconsolidation_safely(self, task: dict) -> None:
        """Wrap the per-task reconsolidation handler with logging + exception swallowing."""
        try:
            await self._run_reconsolidation(task)
        except Exception as exc:
            logger.exception("[s2] reconsolidation failed: %s", exc)

    async def _run_reconsolidation(self, task: dict) -> None:
        """Refresh the last-reactivated timestamp for recalled nodes without extra inference."""
        payload = task.get("payload") or {}
        query = payload.get("query") or ""
        node_ids = payload.get("node_ids") or []
        if not node_ids:
            return

        now = int(time.time())
        for nid in node_ids:
            node = self.factory.vector.get(nid)
            if node is None:
                continue
            custom = dict(node.custom or {})
            custom["last_reactivated_at"] = now
            node.custom = custom
            self.factory.vector.upsert([node])

        logger.info(
            "reconsolidation user=%s n_nodes=%d",
            task.get("user_id", ""),
            len(node_ids),
        )

        try:
            self.factory.cache.log_pipeline(
                request_id=f"reconsolidation::{task.get('user_id', '')}",
                stage="RECONSOLIDATION_DRAIN",
                payload={
                    "query": query[:120],
                    "n_nodes": len(node_ids),
                },
            )
        except Exception:
            pass

    def _ensure_scheduled_loop(self) -> None:
        """Start the periodic background loop the first time scheduled mode is used."""
        if self._scheduled_started:
            return
        self._scheduled_started = True
        interval = self.factory.settings.system2_schedule_interval_sec
        self._scheduled_task = asyncio.create_task(self._scheduled_loop(interval))
        self._scheduled_task.add_done_callback(_log_task_exception)

    async def _scheduled_loop(self, interval: int) -> None:
        """Periodically drain every pending S2 task (cognition + reconsolidation)."""
        while True:
            try:
                await asyncio.sleep(interval)
                # Drain everything queued; both cognition and reconsolidation handlers
                # honour their own per-user serialization where applicable.
                await self._digest_pending()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("[s2] scheduled loop tick failed: %s", exc)


def _log_task_exception(task: asyncio.Task) -> None:
    """Surface exceptions from background tasks instead of letting them die silently."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.exception("[s2] background task raised: %s", exc)
