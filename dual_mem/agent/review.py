"""Digest-time review of decisions that were deferred at write time.

Two kinds of items are queued by ``MemAgent``:

- ``relation``: a memory written as an independent SUPPLEMENT because Jev answered
  NEED_CONTEXT, was unsure between several targets, or chose a relation below
  ``reconcile_review_confidence``. Review re-decides it against the scope's current
  material (which may now contain more evidence) and accepts that decision.
- ``extract``: an L0 source whose extraction Jev judged incomplete. Review re-runs
  extraction and decisions; memories already committed from it are SKIPped as known.

A failing item is retried on later digests and quarantined after
``review_max_attempts``; it never blocks other items or scopes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.agent.reconciler import Reconciler
from dual_mem.errors import DecisionContextError
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryNode

logger = logging.getLogger("dual_mem.agent.review")


class ReviewWorker:
    def __init__(self, *, factory: ComponentFactory) -> None:
        self.factory = factory
        self.agent = MemAgent(factory=factory)
        settings = factory.settings
        # A deferred low-confidence decision is accepted on review.
        self.reconciler = Reconciler(
            jev=factory.jev,
            llm=factory.llm,
            embed=factory.embed,
            vector=factory.vector,
            candidate_limit=settings.reconcile_candidate_limit,
            batch_size=settings.reconcile_batch_size,
            max_state_chars=settings.jev_max_question_chars,
            review_confidence=0.0,
        )

    async def run(self) -> dict[str, int]:
        """Review every pending item; returns counts and the scopes that changed."""
        items = self.factory.cache.list_reviews()
        by_scope: dict[tuple[str, str, str], list[dict]] = {}
        for item in items:
            by_scope.setdefault(
                (item["app_id"], item["user_id"], item["agent_id"]), []
            ).append(item)
        sem = asyncio.Semaphore(self.factory.settings.reconcile_concurrency)
        stats = {"reviewed": 0, "failed": 0}
        changed: set[tuple[str, str, str]] = set()

        async def process(scope: tuple[str, str, str], queue: list[dict]) -> None:
            async with sem:
                # Same-scope items share evolution chains: keep them ordered.
                for item in queue:
                    try:
                        if await self._review(item):
                            changed.add(scope)
                    except (DecisionContextError, ValueError) as exc:
                        stats["failed"] += 1
                        logger.warning(
                            "review %s %s failed: %s", item["kind"], item["node_id"], exc
                        )
                        self.factory.cache.fail_review(
                            item["id"],
                            error=str(exc),
                            max_attempts=self.factory.settings.review_max_attempts,
                        )
                        continue
                    self.factory.cache.resolve_review(item["id"])
                    stats["reviewed"] += 1

        await asyncio.gather(*(process(s, q) for s, q in by_scope.items()))
        if self.factory.settings.mode == "dual":
            for app_id, user_id, agent_id in changed:
                self.factory.cache.enqueue_s2_task(
                    user_id=user_id, app_id=app_id, agent_id=agent_id
                )
        return stats

    async def _review(self, item: dict) -> bool:
        if item["kind"] == "extract":
            return await self._review_extract(item)
        return await self._review_relation(item)

    async def _review_extract(self, item: dict) -> bool:
        raw = self.factory.vector.get(item["node_id"])
        if raw is None or raw.layer != Layer.L0_RAW:
            return False
        origin = item["payload"]["request_key"]
        done = self.factory.vector.completed_request(origin)
        if done is None:
            # Queued before the write that degraded; re-extracting now would write
            # a second set next to it. The write re-queues this item on commit.
            raise DecisionContextError(
                "source write has not completed (in flight or failed); review waits"
            )
        if raw.node_id not in done["needs_review"]:
            # A retry of that write completed with faithful coverage.
            return False
        # Stable per item and originating write: a crash after the commit but before
        # resolve_review is recovered from the plan journal instead of re-decided.
        ids, _, _ = await self.agent.run_system1(
            content=raw.content,
            raw_node=raw,
            request_id=f"review:{item['id']}:{origin}",
            review_attempt=True,
        )
        return bool(ids)

    async def _review_relation(self, item: dict) -> bool:
        node = self.factory.vector.get(item["node_id"])
        if node is None:
            request_key = item["payload"].get("request_key")
            if request_key and self.factory.vector.completed_request(request_key):
                # The write committed after replanning without this version.
                return False
            # Queued before its plan: still in flight, or the write failed.
            # A later commit re-queues it; otherwise it ends up quarantined.
            raise DecisionContextError(
                "provisional version is not committed (write in flight or failed)"
            )
        if not node.is_latest or not (node.custom or {}).get("needs_review"):
            # Superseded/archived since, or committed by a plan that did not defer.
            return False
        source_id = (node.custom or {}).get("source_node_id") or item["payload"].get(
            "source_id"
        )
        source = self.factory.vector.get(source_id) if source_id else None
        scope = (node.app_id, node.user_id, node.agent_id)
        current_time = datetime.fromtimestamp(
            node.memory_at or node.gmt_created, tz=timezone.utc
        ).isoformat()
        (op,) = await self.reconciler.reconcile(
            new_memories=[node.content],
            new_memories_meta=[
                {
                    "layer": node.layer.value,
                    "kind": node.kind,
                    "tags": node.tags,
                    "source_ids": (node.custom or {}).get("source_ids", []),
                    "valid_from": node.valid_from,
                    "valid_until": node.valid_until,
                    "temporal_scope": (node.custom or {}).get("temporal_scope"),
                }
            ],
            app_id=node.app_id,
            user_id=node.user_id,
            agent_id=node.agent_id,
            current_time=current_time,
            exclude_ids=[node.node_id],
            source_content=source.content if source else "",
            memory_at=node.memory_at,
            check_coverage=False,
        )
        if op.needs_review:
            raise DecisionContextError(f"review still undecided: {op.review_reason}")
        resolved_custom = {
            k: v
            for k, v in (node.custom or {}).items()
            if k not in ("needs_review", "review_reason", "review_targets")
        }
        if op.op == "ADD" and op.update_type in ("SUPPLEMENT", "TEMPORAL"):
            # The provisional independent append was correct; clear the flag.
            await asyncio.to_thread(
                self.factory.vector.update_payload,
                node.node_id,
                {"custom": json.dumps(resolved_custom, ensure_ascii=False)},
            )
            return True
        archives: list[str] = []
        nodes: list[MemoryNode] = []
        expected: dict[str, int] = {node.node_id: node.revision}
        if op.op in ("SKIP", "RAW_ONLY", "INVALID"):
            archives = [node.node_id]
        elif op.op == "DELETE":
            _, expected_targets, archives = await self.agent._prepare_nodes(
                [op], source or node, f"review:{item['id']}"
            )
            expected.update(expected_targets)
            # The retraction itself carries no further state.
            archives = [*archives, node.node_id]
        else:
            nodes, expected_targets, archives = await self.agent._prepare_nodes(
                [op], source or node, f"review:{item['id']}"
            )
            expected.update(expected_targets)
            # The resolved version replaces the provisional one.
            for new in nodes:
                new.supersedes = [*new.supersedes, node.node_id]
                new.session_id, new.memory_at = node.session_id, node.memory_at
        if not await asyncio.to_thread(
            self.factory.vector.apply_version_plan,
            nodes=nodes,
            expected=expected,
            archives=archives,
            scope=scope,
            expected_epoch=op.scope_epoch,
            request_key=f"review:{item['id']}",
            decisions=[
                d.to_dict() for d in MemAgent.decisions([op], nodes)
            ],
        ):
            raise DecisionContextError("review targets changed; retried on next digest")
        return True


__all__ = ["ReviewWorker"]
