# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async reconciler worker that drains queued reconcile tasks (one task per write),
loads the freshly written nodes, runs the LLM-driven Reconciler to emit ADD/SUPERSEDE/DELETE
ops, and applies them. Same-user tasks must be serialized (caller holds a per-user lock) to
avoid racing on the same evolution chain.
"""
import asyncio
import logging

from dual_mem.agent.reconciler import ReconcileOp, Reconciler
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.system2.reconcile")


def _norm_content(text: str | None) -> str:
    """Normalize content for coverage matching: strip + lowercase + collapse whitespace."""
    return " ".join((text or "").split()).lower()


def _reflect_custom(op: ReconcileOp) -> dict:
    """Pack reflector metadata (update_type, temporal_scope, negation) into node.custom."""
    out: dict = {}
    if op.update_type:
        out["update_type"] = op.update_type
    if op.temporal_scope:
        out["temporal_scope"] = op.temporal_scope
    if op.negation:
        out["negation"] = True
    return out


class ReconcilerWorker:
    """Drains the reconcile_queue and merges freshly written nodes into the existing chain."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory
        llm = factory.llm
        if llm is None:
            raise RuntimeError("ReconcilerWorker requires factory.llm (system1/dual mode)")
        self.reconciler = Reconciler(
            llm=llm,
            embed=factory.embed,
            vector=factory.vector,
            enable_search_query=factory.settings.reconcile_search_query,
            policy=factory.settings.reconcile_policy,
            weak_candidate_score=factory.settings.reconcile_weak_candidate_score,
        )

    async def reconcile_pending(self, *, app_id: str, user_id: str, agent_id: str = "") -> int:
        """Process all pending reconcile tasks for one (app, user, agent) triple; return count.

        Tasks are drained then processed with bounded concurrency (``reconcile_concurrency``).
        Each task's bottleneck is one large-prompt LLM call inside ``reconcile()`` (recall +
        LLM is read-only; apply is a fast vector upsert), so running several concurrently
        collapses an otherwise serial chain of ~9s LLM round-trips. ``reconcile_concurrency=1``
        preserves the original strict-serial behaviour.
        """
        cache = self.factory.cache
        tasks: list[dict] = []
        while True:
            task = cache.dequeue_reconcile_task(
                app_id=app_id, user_id=user_id, agent_id=agent_id
            )
            if task is None:
                break
            tasks.append(task)
        if not tasks:
            return 0

        concurrency = max(1, self.factory.settings.reconcile_concurrency)
        if concurrency == 1 or len(tasks) == 1:
            processed = 0
            for task in tasks:
                if await self._process_task_safely(task):
                    processed += 1
            return processed

        sem = asyncio.Semaphore(concurrency)

        async def _guarded(task: dict) -> bool:
            async with sem:
                return await self._process_task_safely(task)

        results = await asyncio.gather(*[_guarded(t) for t in tasks])
        return sum(1 for ok in results if ok)

    async def _process_task_safely(self, task: dict) -> bool:
        """Run one reconcile task, swallowing failures so siblings still drain."""
        try:
            await self._process_task(task)
            return True
        except Exception as exc:
            logger.exception("[reconcile] task failed: %s", exc)
            return False

    async def _process_task(self, task: dict) -> None:
        """Reconcile the freshly written node ids carried by a single task."""
        node_ids: list[str] = task.get("node_ids") or []
        if not node_ids:
            return
        app_id = task["app_id"]
        user_id = task["user_id"]
        agent_id = task.get("agent_id") or ""

        new_memories: list[str] = []
        new_meta: list[dict] = []
        nodes_to_remove: list[str] = []
        for nid in node_ids:
            node = self.factory.vector.get(nid)
            if node is None:
                continue
            new_memories.append(node.content)
            new_meta.append(
                {
                    "content": node.content,
                    "layer": node.layer.value,
                    "tags": list(node.tags),
                    "speculate": node.speculate,
                }
            )
            nodes_to_remove.append(nid)

        if not new_memories:
            return

        ops = await self.reconciler.reconcile(
            new_memories=new_memories,
            new_memories_meta=new_meta,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            current_time="",
            exclude_ids=nodes_to_remove,
        )
        logger.info(
            "reconcile user=%s n_new=%d ops=%d supersedes=%d delete=%d",
            user_id, len(new_memories), len(ops),
            sum(1 for op in ops if op.op == "ADD" and op.supersedes),
            sum(1 for op in ops if op.op == "DELETE"),
        )
        try:
            self.factory.cache.log_pipeline(
                request_id=task.get("request_id") or f"reconcile::{user_id}",
                stage="RECONCILE",
                payload={
                    "ops_count": len(ops),
                    "supersedes_count": sum(1 for op in ops if op.op == "ADD" and op.supersedes),
                    "delete_count": sum(1 for op in ops if op.op == "DELETE"),
                    "update_types": [op.update_type for op in ops if op.op == "ADD"],
                },
            )
        except Exception:
            pass
        if not ops:
            return

        await self._apply_ops(
            ops,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id="",
        )

        # Shadow ONLY the fast-write originals whose content a reconcile ADD re-emitted
        # (i.e. were actually re-written). Originals not covered by any ADD stay ACTIVE —
        # never blanket-shadow on "ops exist", or a merge/skip that returns fewer ADDs than
        # originals silently loses facts (the LME "数娃漏 1" failure mode).
        added_contents = {
            _norm_content(op.content)
            for op in ops
            if op.op != "DELETE" and op.content
        }
        self._shadow_covered_originals(
            nodes_to_remove, added_contents, user_id=user_id
        )

    def _shadow_covered_originals(
        self, node_ids: list[str], added_contents: set[str], *, user_id: str
    ) -> None:
        """Soft-remove only fast-write originals whose content was re-emitted by a reconcile ADD."""
        if not added_contents:
            return
        for nid in node_ids:
            node = self.factory.vector.get(nid)
            if node is None:
                continue
            if _norm_content(node.content) not in added_contents:
                continue
            old_meta = node.to_metadata()
            node.status = MemoryStatus.SHADOW
            node.is_latest = False
            self.factory.vector.upsert([node])
            self.factory.history.append(
                event="DELETE",
                node_id=node.node_id,
                user_id=user_id,
                old=old_meta,
                new=node.to_metadata(),
            )

    async def _apply_ops(
        self,
        ops,
        *,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> None:
        """Apply reconcile ops (ADD/SUPERSEDE/DELETE) to the vector store + history log.

        Embeddings for every ADD/SUPERSEDE op are computed up front in a single
        ``embed_batch`` round-trip, instead of one serial ``embed_queued`` await per
        op. Serial ``embed_queued`` in a loop is actively harmful here: the queue only
        ever holds one item, so each call eats the full batch-window wait plus a
        batch-of-one RTT — N ops become N serial round-trips. One ``embed_batch`` call
        collapses that to a single round-trip.
        """
        add_ops = [op for op in ops if op.op != "DELETE"]
        embeddings: list[list[float]] = []
        if add_ops:
            embeddings = await self.factory.embed.embed_batch(
                [op.content or "" for op in add_ops]
            )
        emb_iter = iter(embeddings)

        for op in ops:
            if op.op == "DELETE":
                target = self.factory.vector.get(op.memory_id) if op.memory_id else None
                if target is None:
                    continue
                old_meta = target.to_metadata()
                target.status = MemoryStatus.SHADOW
                target.is_latest = False
                self.factory.vector.upsert([target])
                self.factory.history.append(
                    event="DELETE",
                    node_id=target.node_id,
                    user_id=user_id,
                    old=old_meta,
                    new=target.to_metadata(),
                )
                continue

            try:
                layer = Layer(op.layer.upper()) if op.layer else Layer.L2_FACT
            except ValueError:
                layer = Layer.L2_FACT
            new_node = MemoryNode(
                content=op.content or "",
                layer=layer,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                tags=list(op.tags),
                status=MemoryStatus.ACTIVE,
                is_latest=True,
                supersedes=list(op.supersedes),
                custom=_reflect_custom(op) or None,
            )
            new_node.embedding = next(emb_iter)
            self.factory.vector.upsert([new_node])
            self.factory.history.append(
                event="SUPERSEDE" if op.supersedes else "ADD",
                node_id=new_node.node_id,
                user_id=user_id,
                old=None,
                new=new_node.to_metadata(),
            )

            for old_id in op.supersedes:
                self.factory.vector.mark_superseded(
                    old_id, superseded_by_id=new_node.node_id
                )
