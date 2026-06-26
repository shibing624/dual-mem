# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: dual-mode writer that wraps the System1 writer and dispatches asynchronous
System2 cognition. Supports three trigger modes (per_write / manual / scheduled) with a
per-user lock so concurrent writes for the same user serialize through reconcile -> S2
agent -> cross-domain sweeper, while different users run in parallel. Reconsolidation
tasks (enqueued by the read-side hook) drain through ``_run_reconsolidation`` which marks
recalled nodes as reactivated and detects emotional valence shifts vs the query.
"""
import asyncio
import logging
import time

from dual_mem.agent.gate import AttentionalGate
from dual_mem.locks import LockRegistry
from dual_mem.registry import ComponentFactory
from dual_mem.system2.cross_domain_sweeper import CrossDomainSweeper
from dual_mem.system2.reconciler_worker import ReconcilerWorker
from dual_mem.system2.system2_agent import System2Agent
from dual_mem.writer.memory_writer import MemoryWriter, WriterOutcome

logger = logging.getLogger("dual_mem.system2.writer")

_PER_WRITE = "per_write"
_SCHEDULED = "scheduled"

# Significant valence shift triggers a reactivation flag (the user "re-feels" a memory).
_REACTIVATION_VALENCE_DELTA = 0.4


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
        user_queries: list[str] | None = None,
        agent_context: str | None = None,
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
            user_queries=user_queries,
            agent_context=agent_context,
        )

        # Always enqueue an S2 cognition task; reconcile_queue was already populated
        # inside MemAgent.run when fast-write happened.
        self.factory.cache.enqueue_s2_task(user_id=user_id, app_id=app_id, agent_id=agent_id)

        mode = self.factory.settings.system2_trigger_mode
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
        while True:
            task = cache.dequeue_s2_task()
            if task is None:
                break
            if task.get("task_type", "cognition") == "reconsolidation":
                await self._run_reconsolidation_safely(task)
                self.last_digest_stats["reconsolidation_tasks"] += 1
            else:
                await self._digest_user_safely(
                    app_id=task["app_id"],
                    user_id=task["user_id"],
                    agent_id=task.get("agent_id") or "",
                )
                self.processed_pairs.add((task["app_id"], task["user_id"]))
            processed += 1
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
        """Mark recalled nodes as reactivated; flag emotional shift vs the recall query.

        Strategy (zero extra LLM cost): score the query through the existing AttentionalGate
        heuristic, compare its arousal/valence with each recalled node's stored emotion in
        ``custom``. Significant valence delta → set ``custom.reactivation = True`` and
        ``custom.reactivation_at`` (timestamp). Always bumps ``custom.last_reactivated_at``.
        """
        payload = task.get("payload") or {}
        query = payload.get("query") or ""
        node_ids = payload.get("node_ids") or []
        if not node_ids:
            return

        gate = AttentionalGate(threshold=0.0, llm=None)
        gate_result = await gate.evaluate(content=query)
        # Heuristic: gate emotional_arousal + valence inferred from gate result. The gate
        # itself does not return signed valence, but biographical_relevance × arousal sign
        # is a decent proxy. We err on the side of caution and only flag when arousal high.
        query_arousal = float(gate_result.emotional_arousal)

        flagged = 0
        now = int(time.time())
        for nid in node_ids:
            node = self.factory.vector.get(nid)
            if node is None:
                continue
            custom = dict(node.custom or {})
            stored_arousal = float(custom.get("emotional_arousal", 0.0) or 0.0)
            arousal_delta = abs(query_arousal - stored_arousal)
            custom["last_reactivated_at"] = now
            if arousal_delta >= _REACTIVATION_VALENCE_DELTA:
                custom["reactivation"] = True
                custom["reactivation_at"] = now
                flagged += 1
            node.custom = custom
            self.factory.vector.upsert([node])

        logger.info(
            "reconsolidation user=%s n_nodes=%d flagged=%d",
            task.get("user_id", ""), len(node_ids), flagged,
        )

        try:
            self.factory.cache.log_pipeline(
                request_id=f"reconsolidation::{task.get('user_id', '')}",
                stage="RECONSOLIDATION_DRAIN",
                payload={
                    "query": query[:120],
                    "n_nodes": len(node_ids),
                    "flagged_reactivation": flagged,
                    "query_arousal": round(query_arousal, 3),
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
