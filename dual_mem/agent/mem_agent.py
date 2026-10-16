"""One System1 pipeline: generate candidates, Jev decisions, append versions."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import date, datetime, time, timezone
from typing import Any

from dual_mem.agent.extractor import Extractor
from dual_mem.agent.reconciler import ReconcileOp, Reconciler, build_apply_custom
from dual_mem.errors import (
    CoverageError,
    DecisionContextError,
    DecisionPlanError,
    StaleDecisionError,
)
from dual_mem.registry import ComponentFactory
from dual_mem.sdk_models import ChatMessage, CommitResult, MemoryDecision
from dual_mem.types import Layer, MemoryNode

logger = logging.getLogger("dual_mem.agent.mem")

_COVERAGE_FEEDBACK = (
    "the candidates did not faithfully cover every explicit fact, update, number and "
    "independent event in the source, or contained unsupported content"
)
_EMPTY_FEEDBACK = (
    "extraction returned no candidates although the source contains durable facts"
)


def _validity(value: Any, *, end_of_day: bool) -> int | str | None:
    """Resolve an extracted date-only bound to UTC epoch seconds.

    Timestamps with an explicit timezone pass through for the reconciler to parse.
    A malformed bound is dropped (the memory is still written without it).
    """
    if value is None or value == "":
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            day = date.fromisoformat(value)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
            if parsed is not None and parsed.tzinfo is not None:
                return value
            if parsed is not None:
                return int(parsed.replace(tzinfo=timezone.utc).timestamp())
        else:
            moment = time.max if end_of_day else time.min
            return int(
                datetime.combine(day, moment, tzinfo=timezone.utc).timestamp()
            )
    logger.warning("extract: dropping malformed validity bound %r", value)
    return None


class MemAgent:
    def __init__(self, *, factory: ComponentFactory) -> None:
        self.factory = factory
        settings = factory.settings
        self.extractor = Extractor(
            llm=factory.llm,
            max_content_chars=int(
                settings.extract_max_content_tokens * settings.chars_per_token
            ),
            retry_on_failure=settings.extract_retry_on_failure,
            few_shot_enabled=settings.extract_few_shot_enabled,
        )
        self.reconciler = Reconciler(
            jev=factory.jev,
            llm=factory.llm,
            embed=factory.embed,
            vector=factory.vector,
            candidate_limit=settings.reconcile_candidate_limit,
            batch_size=settings.reconcile_batch_size,
            max_state_chars=settings.jev_max_question_chars,
            review_confidence=settings.reconcile_review_confidence,
        )

    @staticmethod
    def _collect_new_memories(
        extracted: dict, source_ids: list[str]
    ) -> tuple[list[str], list[dict]]:
        meta: list[dict[str, Any]] = []
        for key, layer, kind in (
            ("facts", Layer.L1_FACT, "fact"),
            ("identity", Layer.L2_PROFILE, "profile"),
            ("intentions", Layer.L1_FACT, "intention"),
        ):
            for item in extracted.get(key) or []:
                if isinstance(item, str):
                    item = {"content": item}
                if (
                    not isinstance(item, dict)
                    or not str(item.get("content") or "").strip()
                ):
                    continue
                entry = {
                    **item,
                    "content": str(item["content"]).strip(),
                    "layer": layer.value,
                    "kind": kind,
                    "source_ids": list(source_ids),
                    "valid_from": _validity(item.get("valid_from"), end_of_day=False),
                    "valid_until": _validity(item.get("valid_until"), end_of_day=True),
                }
                if not entry.get("temporal_scope"):
                    entry.pop("temporal_scope", None)
                meta.append(entry)
        # Structured basic profile fields are candidates, not an independent KV overwrite rule.
        for key, value in (extracted.get("basic_info") or {}).items():
            if value is not None and str(value).strip():
                meta.append(
                    {
                        "content": f"The user's {key} is {value}.",
                        "layer": Layer.L2_PROFILE.value,
                        "kind": "profile",
                        "source_ids": list(source_ids),
                        "dimension": str(key),
                    }
                )
        return [m["content"] for m in meta], meta

    async def _needs_reextract(self, *, content: str, extracted: dict) -> tuple[bool, float | None]:
        verdict = await self.factory.jev.evaluate(
            state={"content": content, "extracted": extracted},
            questions={
                "value": {
                    "type": "choice",
                    "instructions": "Does this source contain durable facts, profiles, events or explicit intentions worth extracting? Extraction returned no candidates.",
                    "criteria": {
                        "RAW_ONLY": "No durable increment; retain raw evidence",
                        "REEXTRACT": "Meaningful facts were missed; do not silently drop them",
                    },
                }
            },
        )
        return verdict["value"].choice != "RAW_ONLY", verdict["value"].confidence

    @staticmethod
    def request_key(raw: MemoryNode, request_id: str) -> str:
        return f"write:{raw.node_id}:{request_id}"

    def _queue_source_review(self, raw: MemoryNode, *, reason: str, request_id: str) -> None:
        """Queue re-extraction of ``raw``; it runs only after ``request_key`` completes.

        Queued before the write commits (crash safety) and again after it commits
        (reopens an item a concurrent digest deferred while the write was in flight).
        """
        self.factory.cache.enqueue_review(
            kind="extract",
            app_id=raw.app_id,
            user_id=raw.user_id,
            agent_id=raw.agent_id,
            node_id=raw.node_id,
            payload={
                "reason": reason,
                "request_key": self.request_key(raw, request_id),
            },
        )

    async def _complete_without_plan(
        self, raw: MemoryNode, request_id: str, commit: CommitResult
    ) -> None:
        """Record a write that changed no version, so replays and reviews see it done."""
        await asyncio.to_thread(
            self.factory.vector.apply_version_plan,
            nodes=[],
            expected={},
            archives=[],
            scope=(raw.app_id, raw.user_id, raw.agent_id),
            request_key=self.request_key(raw, request_id),
            decisions=[d.to_dict() for d in commit.decisions],
            needs_review=list(commit.needs_review),
        )

    def recover(
        self, raw_node: MemoryNode, request_id: str, *, review_attempt: bool = False
    ) -> tuple[list[str], CommitResult, bool] | None:
        """Return the result of an already committed write and restore its reviews.

        A process can die after the version plan commits but before its review
        intents or result are recorded; the plan journal holds both.
        """
        request_key = self.request_key(raw_node, request_id)
        done = self.factory.vector.completed_request(request_key)
        if done is None:
            return None
        decisions = [MemoryDecision(**d) for d in done["decisions"]]
        nodes = self.factory.vector.get_by_ids(done["extra_node_ids"])
        for node_id in done["needs_review"]:
            if node_id == raw_node.node_id:
                if not review_attempt:
                    self._queue_source_review(
                        raw_node,
                        reason=next(d.review_reason for d in decisions if d.review_reason),
                        request_id=request_id,
                    )
                continue
            node = nodes.get(node_id)
            if node is None or not (node.custom or {}).get("needs_review"):
                continue
            self.factory.cache.enqueue_review(
                kind="relation",
                app_id=node.app_id,
                user_id=node.user_id,
                agent_id=node.agent_id,
                node_id=node_id,
                payload={
                    "reason": node.custom.get("review_reason", ""),
                    "candidates": node.custom.get("review_targets", []),
                    "source_id": raw_node.node_id,
                    "request_key": request_key,
                },
            )
        return (
            list(done["extra_node_ids"]),
            CommitResult(
                done["commit_passed"],
                "recovered completed write",
                decisions,
                list(done["needs_review"]),
            ),
            bool(decisions)
            and all(d.action == "RAW_ONLY" and not d.review_reason for d in decisions),
        )

    async def run_system1(
        self,
        *,
        content: str,
        raw_node: MemoryNode,
        messages: list[ChatMessage] | None = None,
        request_id: str = "",
        source_node_ids: list[str] | None = None,
        review_attempt: bool = False,
    ) -> tuple[list[str], CommitResult, bool]:
        """Extract, decide and commit; degrade instead of dropping the source.

        Coverage rejections regenerate the extraction with the rejection as feedback.
        When ``extract_coverage_retries`` is exhausted, the last candidates are still
        decided and committed, and the L0 source is queued for review by ``digest()``.
        ``review_attempt`` marks the call made by that review (no further queuing).
        """
        if request_id:
            recovered = self.recover(
                raw_node, request_id, review_attempt=review_attempt
            )
            if recovered is not None:
                return recovered
        current_time = datetime.fromtimestamp(
            raw_node.memory_at or raw_node.gmt_created,
            tz=timezone.utc,
        ).isoformat()
        source_ids = source_node_ids or [raw_node.node_id]
        retries = self.factory.settings.extract_coverage_retries
        feedback = ""
        for attempt in range(retries + 1):
            last = attempt == retries
            extracted = await self.extractor.extract_system1(
                content=content,
                current_time=current_time,
                current_messages=messages,
                feedback=feedback,
            )
            self.factory.cache.log_pipeline(
                request_id=request_id,
                stage="EXTRACT",
                payload={"candidates": extracted, "attempt": attempt},
            )
            memories, meta = self._collect_new_memories(extracted, source_ids)
            if not memories:
                missed, confidence = await self._needs_reextract(
                    content=content, extracted=extracted
                )
                if missed and not last:
                    feedback = _EMPTY_FEEDBACK
                    continue
                pending: list[str] = []
                if missed:
                    if review_attempt:
                        raise DecisionContextError(
                            "Jev found durable information missed by extraction"
                        )
                    pending = [raw_node.node_id]
                commit = CommitResult(
                    False,
                    "Jev: missed durable information; queued for review"
                    if missed
                    else "Jev: raw evidence only",
                    [
                        MemoryDecision(
                            action="RAW_ONLY",
                            source_ids=source_ids,
                            reason=(
                                "REEXTRACT: extraction missed durable information"
                                if missed
                                else "RAW_ONLY: no durable increment"
                            ),
                            confidence=confidence,
                            review_reason="REEXTRACT" if missed else None,
                        )
                    ],
                    pending,
                )
                if missed:
                    self._queue_source_review(
                        raw_node, reason="REEXTRACT", request_id=request_id
                    )
                await self._complete_without_plan(raw_node, request_id, commit)
                if missed:
                    self._queue_source_review(
                        raw_node, reason="REEXTRACT", request_id=request_id
                    )
                return [], commit, not missed
            try:
                return await self._decide_and_commit(
                    memories=memories,
                    meta=meta,
                    content=content,
                    raw_node=raw_node,
                    request_id=request_id,
                    current_time=current_time,
                    check_coverage=True,
                )
            except CoverageError:
                if not last:
                    feedback = _COVERAGE_FEEDBACK
                    continue
                if review_attempt:
                    raise
            logger.warning(
                "write %s: coverage still rejected after %d regenerations; "
                "committing candidates and queuing the source for review",
                raw_node.node_id,
                retries,
            )
            # Queue before committing: a crash after the commit must not lose it.
            self._queue_source_review(
                raw_node, reason="coverage", request_id=request_id
            )
            return await self._decide_and_commit(
                memories=memories,
                meta=meta,
                content=content,
                raw_node=raw_node,
                request_id=request_id,
                current_time=current_time,
                check_coverage=False,
                source_review=True,
            )
        raise AssertionError("unreachable")

    async def _decide_and_commit(
        self,
        *,
        memories: list[str],
        meta: list[dict],
        content: str,
        raw_node: MemoryNode,
        request_id: str,
        current_time: str,
        check_coverage: bool,
        source_review: bool = False,
    ) -> tuple[list[str], CommitResult, bool]:
        scope = (raw_node.app_id, raw_node.user_id, raw_node.agent_id)
        lock = self.factory.commit_locks.get("decision::" + "::".join(scope))
        request_key = self.request_key(raw_node, request_id)
        plan_feedback: dict | None = None
        last_plan_error: DecisionPlanError | None = None

        def _rejected(exc: DecisionPlanError, ops: list[ReconcileOp]) -> dict:
            return {
                "constraint": str(exc),
                "instructions": (
                    "Reconsider the entire batch; every retiring current target may be consumed once. "
                    "Distinct events and current state are separate dimensions. "
                    "Preserve all candidates; choose corresponding earlier proposals for dependent versions when needed. "
                    "An earlier proposal can be a relation target only if its action creates a memory or resolves to a current head. "
                    "Do not guess or change source content. "
                    "SKIP, OVERRIDE, MERGE, NEGATE, CONFLICT and HISTORICAL require a selected current head or materialized earlier proposal; DELETE requires a current head; NONE is not coverage."
                ),
                "decisions": [
                    {
                        "proposals": op.proposal_ids,
                        "action": op.decision.get("action", {}).get("choice", op.op),
                        "update_type": op.update_type,
                        "targets": op.target_ids,
                    }
                    for op in ops
                ],
            }

        async def plan_and_commit() -> (
            tuple[list[ReconcileOp], list[MemoryNode], list[str], bool]
        ):
            nonlocal plan_feedback, last_plan_error
            last_plan_error = None
            try:
                ops = await self.reconciler.reconcile(
                    new_memories=memories,
                    new_memories_meta=meta,
                    app_id=raw_node.app_id,
                    user_id=raw_node.user_id,
                    agent_id=raw_node.agent_id,
                    current_time=current_time,
                    exclude_ids=[raw_node.node_id],
                    source_content=content,
                    plan_feedback=plan_feedback,
                    memory_at=raw_node.memory_at,
                    check_coverage=check_coverage,
                )
            except DecisionPlanError as exc:
                last_plan_error = exc
                plan_feedback = _rejected(exc, [])
                return [], [], [], False
            try:
                nodes, expected, archives = await self._prepare_nodes(
                    ops, raw_node, request_id
                )
            except DecisionPlanError as exc:
                last_plan_error = exc
                plan_feedback = _rejected(exc, ops)
                return ops, [], [], False
            if source_review:
                for op in ops:
                    op.review_reason = op.review_reason or "coverage"
            # Review intents are durable before the version plan; ReviewWorker
            # resolves intents whose node was not committed or not flagged.
            pending = self._queue_relation_reviews(ops, nodes, raw_node, request_key)
            if source_review:
                pending.append(raw_node.node_id)
            committed = await asyncio.to_thread(
                self.factory.vector.apply_version_plan,
                nodes=nodes,
                expected=expected,
                archives=archives,
                scope=scope,
                expected_epoch=ops[0].scope_epoch,
                request_key=request_key,
                decisions=[d.to_dict() for d in self.decisions(ops, nodes)],
                needs_review=pending,
            )
            return ops, nodes, archives, committed

        attempts = self.factory.settings.reconcile_retry_attempts
        for attempt in range(attempts):
            # First attempts overlap model I/O. A final scoped retry avoids a
            # same-user stampede while leaving other scopes concurrent.
            if attempt + 1 == attempts:
                async with lock:
                    ops, nodes, archives, committed = await plan_and_commit()
            else:
                ops, nodes, archives, committed = await plan_and_commit()
            if committed:
                return self._committed(
                    ops, nodes, archives, raw_node, request_id, source_review
                )
        if last_plan_error is not None:
            raise last_plan_error
        raise StaleDecisionError("targets changed concurrently; retry write")

    def _committed(
        self,
        ops: list[ReconcileOp],
        nodes: list[MemoryNode],
        archives: list[str],
        raw_node: MemoryNode,
        request_id: str,
        source_review: bool = False,
    ) -> tuple[list[str], CommitResult, bool]:
        self.factory.cache.log_pipeline(
            request_id=request_id,
            stage="JEV_DECISION",
            payload={
                "passed": bool(nodes or archives),
                "ops": [
                    {
                        "op": op.op,
                        "update_type": op.update_type,
                        "proposal_ids": op.proposal_ids,
                        "decision": op.decision,
                        "review_reason": op.review_reason,
                    }
                    for op in ops
                ],
            },
        )
        # Re-assert the intents queued before the commit: a digest that ran while
        # the plan was in flight may have resolved them as "not committed".
        pending = self._queue_relation_reviews(
            ops, nodes, raw_node, self.request_key(raw_node, request_id)
        )
        if source_review:
            self._queue_source_review(
                raw_node, reason="coverage", request_id=request_id
            )
            pending.append(raw_node.node_id)
        return (
            [n.node_id for n in nodes],
            CommitResult(
                bool(nodes or archives),
                "Jev relation decisions",
                self.decisions(ops, nodes),
                pending,
            ),
            all(op.op == "RAW_ONLY" for op in ops) and not source_review,
        )

    def _queue_relation_reviews(
        self,
        ops: list[ReconcileOp],
        nodes: list[MemoryNode],
        raw: MemoryNode,
        request_key: str,
    ) -> list[str]:
        by_pid = {
            pid: node.node_id
            for node in nodes
            for pid in (node.custom or {}).get("proposal_ids", [])
        }
        pending: list[str] = []
        for op in ops:
            if not op.needs_review or op.op != "ADD":
                continue
            node_id = by_pid[op.proposal_ids[0]]
            self.factory.cache.enqueue_review(
                kind="relation",
                app_id=raw.app_id,
                user_id=raw.user_id,
                agent_id=raw.agent_id,
                node_id=node_id,
                payload={
                    "reason": op.review_reason,
                    "candidates": op.review_targets,
                    "source_id": raw.node_id,
                    "request_key": request_key,
                },
            )
            pending.append(node_id)
        return pending

    @staticmethod
    def decisions(
        ops: list[ReconcileOp], nodes: list[MemoryNode]
    ) -> list[MemoryDecision]:
        resolved = {
            pid: node.node_id
            for node in nodes
            for pid in (node.custom or {}).get("proposal_ids", [])
        }
        for op in ops:
            if op.op == "SKIP" and op.memory_id:
                for pid in op.proposal_ids:
                    resolved[pid] = resolved.get(op.memory_id, op.memory_id)
        return [
            MemoryDecision(
                action=op.op,
                update_type=op.update_type or None,
                targets=[resolved.get(nid, nid) for nid in op.target_ids],
                proposal_ids=op.proposal_ids,
                source_ids=op.source_ids,
                reason=op.reason,
                confidence=op.decision.get("action", {}).get("confidence"),
                review_reason=op.review_reason or None,
            )
            for op in ops
        ]

    async def _prepare_nodes(
        self,
        ops: list[ReconcileOp],
        raw: MemoryNode,
        request_id: str,
    ) -> tuple[list[MemoryNode], dict[str, int], list[str]]:
        adds = [op for op in ops if op.op == "ADD"]
        ids = {
            pid: str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{raw.node_id}:{request_id}:{pid}")
            )
            for op in adds
            for pid in op.proposal_ids
        }
        proposal_ids = {pid for op in ops for pid in op.proposal_ids}
        expected: dict[str, int] = {}
        archives: list[str] = []
        missing = [op for op in adds if op.embedding is None]
        fresh = (
            await self.factory.embed.embed_batch([op.content for op in missing])
            if missing
            else []
        )
        for op, vector in zip(missing, fresh):
            op.embedding = vector
        nodes: list[MemoryNode] = []
        # A retired target may be consumed by exactly one operation; HISTORICAL
        # only links behind a head that stays current, so several may share it.
        retired: set[str] = set()
        linked: set[str] = set()
        resolved = dict(ids)
        for op in ops:
            if op.op == "SKIP" and op.memory_id:
                for pid in op.proposal_ids:
                    resolved[pid] = resolved.get(op.memory_id, op.memory_id)
        for op in ops:
            unresolved = [
                target
                for target in op.target_ids
                if target in proposal_ids and target not in resolved
            ]
            if unresolved:
                raise DecisionPlanError(
                    "operation targets an earlier proposal that produced no memory"
                )
            if op.op == "DELETE" and any(
                target in proposal_ids for target in op.target_ids
            ):
                raise DecisionContextError(
                    "DELETE cannot target an uncommitted proposal"
                )
            historical = op.update_type == "HISTORICAL"
            targets = [
                resolved.get(nid, nid)
                for nid in (op.target_ids if historical else op.supersedes)
            ]
            if retired.intersection(targets) or (
                not historical and linked.intersection(targets)
            ):
                raise DecisionPlanError(
                    "multiple operations consume one target; replan required"
                )
            (linked if historical else retired).update(targets)
            expected.update(op.expected_revisions)
            if op.op == "DELETE":
                archive_targets: list[str] = list(op.target_ids) or (
                    [op.memory_id] if op.memory_id else []
                )
                if not archive_targets or any(
                    nid not in op.expected_revisions for nid in archive_targets
                ):
                    raise DecisionContextError(
                        "archive target is not a scoped current node"
                    )
                if (retired | linked).intersection(archive_targets):
                    raise DecisionPlanError(
                        "multiple operations consume one target; replan required"
                    )
                retired.update(archive_targets)
                archives.extend(archive_targets)
            elif op.op == "ADD":
                custom = build_apply_custom(
                    op,
                    vector=self.factory.vector,
                    new_memory_at=raw.memory_at,
                    source_node_id=raw.node_id,
                )
                custom["source_ids"] = op.source_ids
                custom["conflicts_with"] = [
                    resolved.get(nid, nid) for nid in op.conflicts_with
                ]
                nodes.append(
                    MemoryNode(
                        content=op.content or "",
                        layer=Layer(op.layer or Layer.L1_FACT),
                        kind=op.kind,
                        node_id=ids[op.proposal_ids[0]],
                        app_id=raw.app_id,
                        user_id=raw.user_id,
                        agent_id=raw.agent_id,
                        session_id=raw.session_id,
                        tags=op.tags,
                        memory_at=raw.memory_at,
                        supersedes=[] if historical else targets,
                        superseded_by=targets if historical else [],
                        is_latest=not historical,
                        custom=custom,
                        valid_from=op.valid_from,
                        valid_until=op.valid_until,
                        embedding=op.embedding,
                        owner=raw.owner,
                    )
                )
        # Dependencies are earlier proposal references. Every target must exist before apply.
        all_ids = {n.node_id for n in nodes}
        for node in nodes:
            for nid in [*node.supersedes, *node.superseded_by]:
                if nid not in all_ids and nid not in expected:
                    raise DecisionContextError(
                        "unresolved proposal or target reference"
                    )
        return nodes, expected, archives
