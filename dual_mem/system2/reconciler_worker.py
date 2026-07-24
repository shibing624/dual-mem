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


def _chain_sort_key(node: MemoryNode) -> tuple[int, int]:
    """Chronological ordering key for evolution linking: session memory_at, then created ts."""
    return (
        node.memory_at if node.memory_at is not None else 0,
        node.gmt_created or 0,
    )


def link_evolution_chains_heuristic(
    factory: ComponentFactory, *, app_id: str, user_id: str, agent_id: str = ""
) -> int:
    """Zero-LLM heuristic that back-fills evolution-chain pointers on the skip_llm fast-path.

    On the ``reconcile_skip_llm`` path the Extractor writes every fact/identity node as an
    independent ACTIVE node and reconcile is skipped entirely, so preference updates never get
    ``supersedes``/``superseded_by`` pointers and ``expand_evolution_chains`` has nothing to
    trace (the root cause of PersonaMem ``track_full_preference_evolution`` misses).

    This groups a user's ACTIVE L2/L4 nodes by layer and normalized non-empty tag and,
    within each bucket of 2+
    ordered chronologically, wires ``older <-supersedes- newer`` edges via
    ``vector.link_supersedes(keep_active=True)`` — the "link-but-don't-hide" mode: old nodes stay
    ACTIVE and recallable while the timeline becomes reconstructable. Idempotent: skips edges
    whose pointer already exists. Returns the number of edges created.
    """
    vector = factory.vector
    nodes = vector.get_many(
        {"$and": [{"app_id": app_id}, {"user_id": user_id}]},
        limit=1000,
    )
    candidates = [
        n
        for n in nodes
        if n.layer in (Layer.L2_FACT, Layer.L4_IDENTITY)
        and n.status == MemoryStatus.ACTIVE
        and (n.agent_id or "") == (agent_id or "")
        and n.tags
    ]
    if len(candidates) < 2:
        return 0

    by_tag: dict[tuple[Layer, str], list[MemoryNode]] = {}
    for node in candidates:
        normalized_tags = {tag.strip().lower() for tag in node.tags if tag.strip()}
        for tag in normalized_tags:
            by_tag.setdefault((node.layer, tag), []).append(node)

    edges = 0
    linked_pairs: set[tuple[str, str]] = set()
    for members in by_tag.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=_chain_sort_key)
        for older, newer in zip(ordered, ordered[1:]):
            if older.node_id == newer.node_id:
                continue
            pair = (newer.node_id, older.node_id)
            if pair in linked_pairs:
                continue
            if older.node_id in (newer.supersedes or []):
                linked_pairs.add(pair)
                continue
            if vector.link_supersedes(
                new_id=newer.node_id, old_id=older.node_id, keep_active=True
            ):
                linked_pairs.add(pair)
                edges += 1
    if edges:
        logger.info(
            "[s2] heuristic evolution-linking: created %d chain edge(s) for %s::%s",
            edges,
            app_id,
            user_id,
        )
    return edges


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
            non_destructive=factory.settings.reconcile_non_destructive,
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
        memory_at_by_content: dict[str, int] = {}
        memory_at_candidates: list[int] = []
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
            if node.memory_at is not None:
                memory_at_candidates.append(node.memory_at)
                memory_at_by_content[_norm_content(node.content)] = node.memory_at
        fallback_memory_at = (
            min(memory_at_candidates) if memory_at_candidates else None
        )

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
            memory_at_by_content=memory_at_by_content,
            fallback_memory_at=fallback_memory_at,
        )

        # Shadow ONLY the fast-write originals whose content a reconcile ADD re-emitted
        # (i.e. were actually re-written). Originals not covered by any ADD stay ACTIVE —
        # never blanket-shadow on "ops exist", or a merge/skip that returns fewer ADDs than
        # originals silently loses facts (the LME "数娃漏 1" failure mode).
        #
        # non_destructive guard: when non_destructive=True the contract is "原始 fact 只增
        # 不减"。即使 LLM 把原文 re-emit 成 SUPPLEMENT（content 匹配），也不应 shadow 原始
        # 节点——否则"immutable"保证被打破。skip_llm 路径已完全不跑 _process_task，这里
        # 补 non_destructive+skip_llm=False 的残留路径。
        if self.reconciler.non_destructive:
            logger.debug(
                "non_destructive: skip _shadow_covered_originals (originals stay ACTIVE)"
            )
            return

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

    def _resolve_memory_at(
        self,
        op: ReconcileOp,
        *,
        memory_at_by_content: dict[str, int],
        fallback_memory_at: int | None,
    ) -> int | None:
        """Keep session ``memory_at`` on reconcile ADDs (Hy parity for LME QA dates)."""
        for old_id in op.supersedes:
            old = self.factory.vector.get(old_id)
            if old is not None and old.memory_at is not None:
                return old.memory_at
        matched = memory_at_by_content.get(_norm_content(op.content))
        if matched is not None:
            return matched
        return fallback_memory_at

    async def _apply_ops(
        self,
        ops,
        *,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        memory_at_by_content: dict[str, int] | None = None,
        fallback_memory_at: int | None = None,
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
            memory_at = self._resolve_memory_at(
                op,
                memory_at_by_content=memory_at_by_content or {},
                fallback_memory_at=fallback_memory_at,
            )
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
                memory_at=memory_at,
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
