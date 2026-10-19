"""One System1 pipeline: generate candidates, Jev decisions, append versions."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import date, datetime, time, timezone
from typing import Any

from dual_mem.agent.extractor import Extractor
from dual_mem.agent.reconciler import ReconcileOp, Reconciler, build_apply_custom
from dual_mem.errors import (
    DecisionContextError,
    DecisionPlanError,
    StaleDecisionError,
)
from dual_mem.registry import ComponentFactory
from dual_mem.sdk_models import ChatMessage, CommitResult, MemoryDecision
from dual_mem.types import Layer, MemoryNode

logger = logging.getLogger("dual_mem.agent.mem")


def _confidence(op: ReconcileOp) -> float:
    """Confidence of the action decision; unrecorded confidence ranks lowest."""
    return float((op.decision.get("action") or {}).get("confidence") or 0.0)


def _as_supplement(op: ReconcileOp) -> ReconcileOp:
    """Rewrite one op as an independent append that retires nothing."""
    op.op = "ADD"
    op.update_type = "SUPPLEMENT"
    op.supersedes = []
    op.target_ids = []
    op.conflicts_with = []
    op.negation = False
    op.memory_id = None
    op.expected_revisions = {}
    op.decision["effective"] = "ADD_SUPPLEMENT"
    return op


def _settle_target_claims(
    ops: list[ReconcileOp], *, resolved: dict[str, str]
) -> list[ReconcileOp]:
    """Make one batch executable in a single deterministic pass.

    Two inconsistencies can be produced by one decision round and cannot be fixed by
    asking again (the model has the same information):

    - several proposals retire the same current head — a head may only be consumed
      once, so the highest-confidence claim keeps its relation and the others are
      written as independent memories;
    - a proposal targets an earlier *proposal* of this batch that produced no memory
      (RAW_ONLY / INVALID) — that relation can never resolve, so the dangling target
      is dropped and the proposal keeps whatever other targets it has.

    Resolving here costs one in-memory pass and no model calls; replanning the batch
    costs a full Jev round-trip and usually reproduces the same conflict.
    """
    claims: dict[str, list[int]] = {}
    for idx, op in enumerate(ops):
        if op.op == "DELETE":
            targets = list(op.target_ids) or ([op.memory_id] if op.memory_id else [])
        elif op.op == "ADD" and op.update_type in ("OVERRIDE", "NEGATE"):
            targets = list(op.supersedes)
        else:
            continue
        for target in {resolved.get(t, t) for t in targets}:
            claims.setdefault(target, []).append(idx)

    losers: set[int] = set()
    for target, idxs in claims.items():
        if len(idxs) < 2:
            continue
        ranked = sorted(idxs, key=lambda i: (-_confidence(ops[i]), i))
        for idx in ranked[1:]:
            losers.add(idx)
    if not losers:
        settled = ops
    else:
        settled = []
        for idx, op in enumerate(ops):
            if idx not in losers:
                settled.append(op)
                continue
            logger.warning(
                "proposal %s lost the claim on its target; writing it as an independent memory",
                op.proposal_ids,
            )
            settled.append(_as_supplement(op))

    # An earlier proposal that produced no memory cannot be a relation target.
    proposal_ids = {pid for op in ops for pid in op.proposal_ids}
    has_memory = {
        pid
        for op in settled
        if op.op == "ADD" or (op.op == "SKIP" and op.memory_id)
        for pid in op.proposal_ids
    }
    dead = proposal_ids - has_memory
    for op in settled:
        dangling = [t for t in op.target_ids if t in dead]
        if not dangling:
            continue
        op.target_ids = [t for t in op.target_ids if t not in dead]
        op.supersedes = [t for t in op.supersedes if t not in dead]
        op.conflicts_with = [t for t in op.conflicts_with if t not in dead]
        op.expected_revisions = {
            k: v for k, v in op.expected_revisions.items() if k not in dead
        }
        if op.memory_id in dead:
            op.memory_id = None
        if not op.target_ids and op.op == "DELETE":
            op.op = "RAW_ONLY"
        elif not op.target_ids and (
            op.op == "SKIP" or op.update_type in ("OVERRIDE", "NEGATE")
        ):
            logger.warning(
                "proposal %s targets a proposal without memory; writing it independently",
                op.proposal_ids,
            )
            _as_supplement(op)
    return settled


def _independent(ops: list[ReconcileOp]) -> list[ReconcileOp]:
    """Drop every relation: memories become independent appends, nothing is retired."""
    out: list[ReconcileOp] = []
    for op in ops:
        if op.op in ("DELETE", "SKIP") or (op.op == "ADD" and op.target_ids):
            out.append(_as_supplement(op))
        else:
            out.append(op)
    return out


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
            target_limit=settings.reconcile_target_limit,
            relation_threshold=settings.reconcile_relation_threshold,
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
                if (key == "identity" and isinstance(item, dict) and not item.get("content")
                        and isinstance(item.get("attribute"), str) and item["attribute"].strip()
                        and item.get("value") not in (None, "", [], {})):
                    value = item["value"]
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, ensure_ascii=False)
                    item = dict(item, content=f"The user's {item['attribute']} is {value}.",
                                dimension=item["attribute"])
                if (
                    not isinstance(item, dict)
                    or not str(item.get("content") or "").strip()
                ):
                    continue
                item_layer, item_kind = layer, kind
                labelled = str(item.get("type") or "").strip().lower()
                if key == "facts" and labelled == "profile":
                    item_layer, item_kind = Layer.L2_PROFILE, "profile"
                elif key == "facts" and labelled == "intention":
                    item_layer, item_kind = Layer.L1_FACT, "intention"
                speaker = str(item.get("speaker") or item.get("owner") or "").lower()
                owner = {"user": "user", "assistant": "agent", "agent": "agent"}.get(speaker, "")
                if owner == "agent":
                    item_layer, item_kind = Layer.L1_FACT, "fact"
                entry = {
                    **item,
                    "content": str(item["content"]).strip(),
                    "layer": item_layer.value,
                    "kind": item_kind,
                    "owner": owner,
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

    @staticmethod
    def request_key(raw: MemoryNode, request_id: str) -> str:
        return f"write:{raw.node_id}:{request_id}"

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
        )

    def recover(
        self, raw_node: MemoryNode, request_id: str
    ) -> tuple[list[str], CommitResult, bool] | None:
        """Return the result of an already committed write (plan journal replay)."""
        request_key = self.request_key(raw_node, request_id)
        done = self.factory.vector.completed_request(request_key)
        if done is None:
            return None
        decisions = [MemoryDecision(**d) for d in done["decisions"]]
        return (
            list(done["extra_node_ids"]),
            CommitResult(
                done["commit_passed"],
                "recovered completed write",
                decisions,
            ),
            bool(decisions) and all(d.action == "RAW_ONLY" for d in decisions),
        )

    async def run_system1(
        self,
        *,
        content: str,
        raw_node: MemoryNode,
        messages: list[ChatMessage] | None = None,
        request_id: str = "",
        source_node_ids: list[str] | None = None,
    ) -> tuple[list[str], CommitResult, bool]:
        """Extract, decide and commit.

        Exactly one extraction pass and one decision pass. A coverage rejection or an
        empty extraction is logged and the source stays RAW_ONLY — regenerating inside
        the write path doubles latency for a rare case, and there is no review queue to
        come back to it later.
        """
        if request_id:
            recovered = self.recover(raw_node, request_id)
            if recovered is not None:
                return recovered
        current_time = datetime.fromtimestamp(
            raw_node.memory_at or raw_node.gmt_created,
            tz=timezone.utc,
        ).isoformat()
        source_ids = source_node_ids or [raw_node.node_id]
        extracted = await self.extractor.extract_system1(
            content=content,
            current_time=current_time,
            current_messages=messages,
        )
        self.factory.cache.log_pipeline(
            request_id=request_id,
            stage="EXTRACT",
            payload={"candidates": extracted},
        )
        memories, meta = self._collect_new_memories(extracted, source_ids)
        if not memories:
            # Nothing explicit to store: the raw evidence stays searchable. No model
            # call is spent asking whether the extractor missed something.
            commit = CommitResult(
                False,
                "no explicit candidates; raw evidence only",
                [
                    MemoryDecision(
                        action="RAW_ONLY",
                        source_ids=source_ids,
                        reason="RAW_ONLY: no explicit candidates",
                    )
                ],
            )
            await self._complete_without_plan(raw_node, request_id, commit)
            return [], commit, True
        return await self._decide_and_commit(
            memories=memories,
            meta=meta,
            raw_node=raw_node,
            request_id=request_id,
            current_time=current_time,
        )

    async def _decide_and_commit(
        self,
        *,
        memories: list[str],
        meta: list[dict],
        raw_node: MemoryNode,
        request_id: str,
        current_time: str,
    ) -> tuple[list[str], CommitResult, bool]:
        """One decision pass, one commit.

        The commit checks revisions of the heads the plan actually links (no scope-wide
        epoch: concurrent appends in the same scope do not invalidate each other).
        Internally inconsistent relations become independent appends without another
        decision pass; an externally changed linked head raises StaleDecisionError.
        """
        scope = (raw_node.app_id, raw_node.user_id, raw_node.agent_id)
        request_key = self.request_key(raw_node, request_id)
        # Extraction (the slow LLM part) overlaps freely; only decide+commit is
        # serialized per scope so a concurrent write always sees the previous one's
        # heads. That removes stale plans instead of retrying them.
        async with self.factory.commit_locks.get("decision::" + "::".join(scope)):
            return await self._decide_locked(
                memories=memories,
                meta=meta,
                raw_node=raw_node,
                request_id=request_id,
                current_time=current_time,
                scope=scope,
                request_key=request_key,
            )

    async def _decide_locked(
        self,
        *,
        memories: list[str],
        meta: list[dict],
        raw_node: MemoryNode,
        request_id: str,
        current_time: str,
        scope: tuple[str, str, str],
        request_key: str,
    ) -> tuple[list[str], CommitResult, bool]:
        ops = await self.reconciler.reconcile(
            new_memories=memories,
            new_memories_meta=meta,
            app_id=raw_node.app_id,
            user_id=raw_node.user_id,
            agent_id=raw_node.agent_id,
            current_time=current_time,
            exclude_ids=[raw_node.node_id],
            memory_at=raw_node.memory_at,
        )

        async def commit(plan: list[ReconcileOp]):
            nodes, expected, archives = await self._prepare_nodes(plan, raw_node, request_id)
            ok = await asyncio.to_thread(
                self.factory.vector.apply_version_plan,
                nodes=nodes,
                expected=expected,
                archives=archives,
                scope=scope,
                request_key=request_key,
                decisions=[d.to_dict() for d in self.decisions(plan, nodes)],
            )
            return ok, nodes, archives

        try:
            ok, nodes, archives = await commit(ops)
        except DecisionPlanError as exc:
            logger.warning("write %s: %s; committing relations as independent memories",
                           raw_node.node_id, exc)
            ops = _independent(ops)
            ok, nodes, archives = await commit(ops)
        if not ok:
            # Decide+commit holds the scope lock, so only an external writer (API
            # update/delete) can move a linked head. Surface it; never re-plan.
            raise StaleDecisionError("targets changed concurrently; retry write")
        return self._committed(ops, nodes, archives, raw_node, request_id)

    def _committed(
        self,
        ops: list[ReconcileOp],
        nodes: list[MemoryNode],
        archives: list[str],
        raw_node: MemoryNode,
        request_id: str,
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
                    }
                    for op in ops
                ],
            },
        )
        return (
            [n.node_id for n in nodes],
            CommitResult(
                bool(nodes or archives),
                "Jev relation decisions",
                self.decisions(ops, nodes),
            ),
            all(op.op == "RAW_ONLY" for op in ops),
        )

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
        # Settle in-batch collisions first: it can rewrite ops (relation -> independent
        # append), which changes which proposals need a node id.
        ops = _settle_target_claims(ops, resolved={})
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
                        owner=op.owner or raw.owner,
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
