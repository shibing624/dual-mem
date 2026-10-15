"""One System1 pipeline: generate candidates, Jev decisions, append versions."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from dual_mem.agent.extractor import Extractor
from dual_mem.agent.reconciler import (
    DecisionContextError,
    DecisionPlanError,
    ReconcileOp,
    Reconciler,
    build_apply_custom,
)
from dual_mem.registry import ComponentFactory
from dual_mem.sdk_models import ChatMessage, CommitResult, MemoryDecision
from dual_mem.types import Layer, MemoryNode


class MemAgent:
    def __init__(self, *, factory: ComponentFactory) -> None:
        self.factory = factory
        self.extractor = Extractor(
            llm=factory.llm,
            max_content_chars=int(
                factory.settings.extract_max_content_tokens
                * factory.settings.chars_per_token
            ),
            retry_on_failure=factory.settings.extract_retry_on_failure,
            few_shot_enabled=factory.settings.extract_few_shot_enabled,
        )
        self.reconciler = Reconciler(
            jev=factory.jev,
            llm=factory.llm,
            embed=factory.embed,
            vector=factory.vector,
            candidate_limit=factory.settings.reconcile_candidate_limit,
            batch_size=factory.settings.reconcile_batch_size,
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
                meta.append(
                    {
                        **item,
                        "content": str(item["content"]).strip(),
                        "layer": layer.value,
                        "kind": kind,
                        "source_ids": list(source_ids),
                    }
                )
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

    async def run_system1(
        self,
        *,
        content: str,
        raw_node: MemoryNode,
        messages: list[ChatMessage] | None = None,
        request_id: str = "",
        source_node_ids: list[str] | None = None,
    ) -> tuple[list[str], CommitResult, bool]:
        current_time = datetime.fromtimestamp(
            raw_node.memory_at or raw_node.gmt_created,
            tz=timezone.utc,
        ).isoformat()
        extracted = await self.extractor.extract_system1(
            content=content,
            current_time=current_time,
            current_messages=messages,
        )
        self.factory.cache.log_pipeline(
            request_id=request_id, stage="EXTRACT", payload={"candidates": extracted}
        )
        source_ids = source_node_ids or [raw_node.node_id]
        memories, meta = self._collect_new_memories(extracted, source_ids)
        if not memories:
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
            if verdict["value"].choice != "RAW_ONLY":
                raise DecisionContextError(
                    "Jev found durable information missed by extraction"
                )
            return (
                [],
                CommitResult(
                    False,
                    "Jev: raw evidence only",
                    [
                        MemoryDecision(
                            action="SKIP",
                            source_ids=source_ids,
                            reason="RAW_ONLY: no durable increment",
                            confidence=verdict["value"].confidence,
                        )
                    ],
                ),
                True,
            )
        scope = (raw_node.app_id, raw_node.user_id, raw_node.agent_id)
        lock = self.factory.commit_locks.get("decision::" + "::".join(scope))
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
                    "SKIP, OVERRIDE, MERGE, NEGATE and CONFLICT require a selected current head or materialized earlier proposal; DELETE requires a current head; NONE is not coverage."
                ),
                "decisions": [
                    {
                        "proposals": op.proposal_ids,
                        "action": op.decision.get("action", {}).get(
                            "choice", op.op
                        ),
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
            committed = await asyncio.to_thread(
                self.factory.vector.apply_version_plan,
                nodes=nodes,
                expected=expected,
                archives=archives,
                scope=scope,
                expected_epoch=ops[0].scope_epoch,
                request_key=f"write:{raw_node.node_id}:{request_id}",
                decisions=[
                    decision.to_dict() for decision in self.decisions(ops, nodes)
                ],
            )
            return ops, nodes, archives, committed

        for attempt in range(self.factory.settings.reconcile_retry_attempts):
            # First attempts overlap model I/O. A final scoped retry avoids a
            # same-user stampede while leaving other scopes concurrent.
            if attempt + 1 == self.factory.settings.reconcile_retry_attempts:
                async with lock:
                    ops, nodes, archives, committed = await plan_and_commit()
            else:
                ops, nodes, archives, committed = await plan_and_commit()
            if not committed:
                if attempt + 1 == self.factory.settings.reconcile_retry_attempts:
                    if last_plan_error is not None:
                        raise last_plan_error
                    raise DecisionContextError(
                        "targets changed concurrently; retry write"
                    )
                continue
            self.factory.cache.log_pipeline(
                request_id=request_id,
                stage="COMMIT_DECISION",
                payload={"passed": bool(nodes or archives), "source": "jev"},
            )
            self.factory.cache.log_pipeline(
                request_id=request_id,
                stage="JEV_DECISION",
                payload={
                    "ops": [
                        {
                            "op": op.op,
                            "update_type": op.update_type,
                            "proposal_ids": op.proposal_ids,
                            "decision": op.decision,
                        }
                        for op in ops
                    ]
                },
            )
            return (
                [n.node_id for n in nodes],
                CommitResult(
                    bool(nodes or archives),
                    "Jev relation decisions",
                    self.decisions(ops, nodes),
                ),
                all(op.decision["action"]["choice"] == "RAW_ONLY" for op in ops),
            )
        raise DecisionContextError("write retry budget exhausted")

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
        embeddings = (
            await self.factory.embed.embed_batch([op.content for op in adds])
            if adds
            else []
        )
        nodes: list[MemoryNode] = []
        touched: set[str] = set()
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
            targets = [resolved.get(nid, nid) for nid in op.supersedes]
            if touched.intersection(targets):
                raise DecisionPlanError(
                    "multiple operations consume one target; replan required"
                )
            touched.update(targets)
            expected.update(op.expected_revisions)
            if op.op == "DELETE":
                archive_targets = op.target_ids or (
                    [op.memory_id] if op.memory_id else []
                )
                if not archive_targets or any(
                    nid not in op.expected_revisions for nid in archive_targets
                ):
                    raise DecisionContextError(
                        "archive target is not a scoped current node"
                    )
                if touched.intersection(archive_targets):
                    raise DecisionContextError(
                        "archive and replacement consume the same target"
                    )
                touched.update(archive_targets)
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
                        supersedes=targets,
                        custom=custom,
                        valid_from=op.valid_from,
                        valid_until=op.valid_until,
                        embedding=embeddings[len(nodes)],
                        owner=raw.owner,
                    )
                )
        # Dependencies are earlier proposal references. Every target must exist before apply.
        all_ids = {n.node_id for n in nodes}
        for node in nodes:
            for nid in node.supersedes:
                if nid not in all_ids and nid not in expected:
                    raise DecisionContextError(
                        "unresolved proposal or target reference"
                    )
        return nodes, expected, archives
