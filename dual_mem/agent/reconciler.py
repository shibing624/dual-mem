"""Jev-backed relation decisions; no distance gates or semantic repair rules."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dual_mem.isolation import build_filter
from dual_mem.providers.jev import DecisionProvider
from dual_mem.providers.llm import LLMClient
from dual_mem.storage.vector_store import VectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus

_ACTIONS = {
    "ADD_OVERRIDE": "Same person's explicit CURRENT state change replaces that dimension. 'Moved back/again/now/不再/改为' supplies change evidence, even if an ancestor has the new value. Relocation event itself is a separate SUPPLEMENT; current residence overrides old residence.",
    "ADD_MERGE": "Complementary fragments about ONE object; preserve every number, source and distinct event.",
    "ADD_SUPPLEMENT": "New compatible dimension or distinct completed event. A relocation event is independent of a current-residence profile. Initial current residence with no baseline is SUPPLEMENT, not TEMPORAL.",
    "ADD_TEMPORAL": "Explicit LIMITED exception (this week/temporarily/until a date) coexisting with a long-term baseline. 'Currently/目前/现在' alone is not a limited exception.",
    "ADD_NEGATE": "Explicit termination or denial of an existing claim.",
    "ADD_CONFLICT": "Competing SIMULTANEOUS incompatible claims with no explicit change, correction, ordering or authority. A user's explicit move/change/back is OVERRIDE, not CONFLICT. Different sequential events are SUPPLEMENT, not CONFLICT.",
    "RAW_ONLY": "No durable fact, event, profile or intention; retain source evidence only.",
    "SKIP": "A corresponding current head or earlier proposal fully covers this proposal; NEVER skip because it appears in the new source or history; no changed state, event, number or time scope.",
    "DELETE": "Explicit retraction or redundant representation with a retained canonical source; archive, not erase.",
    "INVALID": "Candidate is unsupported by source, invented or incorrectly attributed; reject extraction.",
    "NEED_CONTEXT": "Evidence is insufficient or inconsistent; never guess based on insertion time.",
}


@dataclass
class ReconcileOp:
    op: str = "ADD"
    content: str | None = None
    layer: str | None = None
    kind: str = "fact"
    supersedes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    memory_id: str | None = None
    reason: str = ""
    update_type: str = ""
    temporal_scope: str | None = None
    negation: bool = False
    new_index: int | None = None
    proposal_ids: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    expected_revisions: dict[str, int] = field(default_factory=dict)
    source_ids: list[str] = field(default_factory=list)
    valid_from: int | None = None
    valid_until: int | None = None
    decision: dict[str, Any] = field(default_factory=dict)
    supersede_reason: str = ""
    scope_epoch: int = 0
    target_ids: list[str] = field(default_factory=list)


class DecisionContextError(RuntimeError):
    """A model requested evidence or returned an inconsistent executable decision."""


class DecisionPlanError(DecisionContextError):
    """An otherwise complete model plan violates execution dependencies."""


def _timestamp(value: Any) -> int | None:
    """Normalize an explicitly supplied epoch or ISO timestamp; never infer dates."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return int(parsed.timestamp())
        except ValueError:
            pass
    raise DecisionContextError(
        "validity requires epoch seconds or an ISO timestamp with timezone"
    )


def collect_merged_timestamps(
    *,
    update_type: str,
    new_memory_at: int | None,
    old_nodes: list[MemoryNode],
    extra: list[int] | None = None,
) -> list[int]:
    if update_type != "MERGE":
        return []
    stamps = [new_memory_at, *(extra or [])]
    for old in old_nodes:
        stamps.append(old.memory_at)
        stamps.extend((old.custom or {}).get("merged_timestamps") or [])
    return sorted({int(ts) for ts in stamps if ts is not None})


def reflect_custom(
    op: ReconcileOp, *, merged_timestamps: list[int] | None = None
) -> dict:
    return {
        key: value
        for key, value in {
            "update_type": op.update_type,
            "temporal_scope": op.temporal_scope,
            "negation": op.negation,
            "merged_timestamps": merged_timestamps,
            "conflicts_with": op.conflicts_with,
            "source_ids": op.source_ids,
            "proposal_ids": op.proposal_ids,
            "decision": op.decision,
            "kind": op.kind,
        }.items()
        if value
    }


def build_apply_custom(
    op: ReconcileOp,
    *,
    vector: VectorStore,
    new_memory_at: int | None,
    extra: list[int] | None = None,
    source_node_id: str = "",
) -> dict:
    old = list(vector.get_by_ids(op.supersedes).values()) if op.supersedes else []
    custom = reflect_custom(
        op,
        merged_timestamps=collect_merged_timestamps(
            update_type=op.update_type,
            new_memory_at=new_memory_at,
            old_nodes=old,
            extra=extra,
        ),
    )
    if source_node_id:
        custom["source_node_id"] = source_node_id
    return custom


class Reconciler:
    """Batch atomic Jev questions, then generate text only for selected MERGEs."""

    def __init__(
        self,
        *,
        jev: DecisionProvider,
        llm: LLMClient,
        embed: Any,
        vector: VectorStore,
        candidate_limit: int = 64,
        batch_size: int = 12,
    ) -> None:
        self.jev, self.llm, self.embed, self.vector = jev, llm, embed, vector
        self.candidate_limit, self.batch_size = candidate_limit, batch_size

    def history(self, head: MemoryNode, scope: tuple[str, str, str]) -> list[dict]:
        found: dict[str, MemoryNode] = {}
        pending = list(head.supersedes)
        while pending:
            missing = [
                nid for nid in pending if nid not in found and nid != head.node_id
            ]
            if not missing:
                break
            fetched = self.vector.get_by_ids(missing)
            if set(fetched) != set(missing):
                raise DecisionContextError(
                    "missing history node; refuse incomplete decision"
                )
            pending = []
            for nid, node in fetched.items():
                if (node.app_id, node.user_id, node.agent_id) != scope:
                    raise DecisionContextError("history crosses tenant scope")
                found[nid] = node
                pending.extend(node.supersedes)
        return [
            self._context_node(node)
            for node in sorted(
                found.values(),
                key=lambda n: n.memory_at or n.gmt_created,
                reverse=True,
            )
        ]

    @staticmethod
    def _context_node(node: MemoryNode) -> dict:
        return {
            "id": node.node_id,
            "content": node.content,
            "layer": node.layer.value,
            "kind": node.kind,
            "is_latest": node.is_latest,
            "revision": node.revision,
            "memory_at": node.memory_at,
            "valid_from": node.valid_from,
            "valid_until": node.valid_until,
            "supersedes": node.supersedes,
            "update_type": (node.custom or {}).get("update_type"),
            "source_ids": (node.custom or {}).get("source_ids", []),
            "owner": node.owner,
            "session_id": node.session_id,
            "conflicts_with": (node.custom or {}).get("conflicts_with", []),
        }

    async def reconcile(
        self,
        *,
        new_memories: list[str],
        new_memories_meta: list[dict],
        app_id: str,
        user_id: str,
        agent_id: str,
        current_time: str,
        exclude_ids: list[str] | None = None,
        source_content: str = "",
        plan_feedback: dict | None = None,
    ) -> list[ReconcileOp]:
        if not new_memories:
            return []
        scope = (app_id, user_id, agent_id)
        scope_epoch = self.vector.scope_epoch(scope)
        where = build_filter(
            app_ids=[app_id],
            user_id=user_id,
            agent_ids=[agent_id],
            layers=[Layer.L1_FACT, Layer.L2_PROFILE, Layer.L3_DERIVED],
            latest_only=True,
            statuses=[MemoryStatus.ACTIVE],
        )
        existing = self.vector.get_many(where, limit=100_001)
        excluded = set(exclude_ids or [])
        existing = [n for n in existing if n.node_id not in excluded]
        if len(existing) > 100_000:
            raise DecisionContextError("scope exceeds material scan budget")
        if len(existing) > self.candidate_limit:
            selected: list[MemoryNode] = []
            for start in range(0, len(existing), self.candidate_limit):
                page = existing[start : start + self.candidate_limit]
                questions = {
                    n.node_id: {
                        "type": "choice",
                        "instructions": f"Is current memory {n.node_id} potentially relevant to any proposal as coverage, an update, conflict or a temporal baseline? Prefer INCLUDE when uncertain.",
                        "criteria": {
                            "INCLUDE": "Potentially relevant",
                            "EXCLUDE": "Clearly unrelated",
                        },
                    }
                    for n in page
                }
                answers = await self.jev.evaluate(
                    state={
                        "current": [self._context_node(n) for n in page],
                        "proposals": new_memories,
                    },
                    questions=questions,
                )
                selected.extend(
                    n for n in page if answers[n.node_id].choice == "INCLUDE"
                )
            if len(selected) > self.candidate_limit:
                raise DecisionContextError(
                    "Jev selected more material than context budget; increase candidate limit"
                )
            existing = selected
        excluded = set(exclude_ids or [])
        existing = [n for n in existing if n.node_id not in excluded]
        current = [
            dict(self._context_node(n), history_versions=self.history(n, scope))
            for n in existing
        ]
        by_id = {n.node_id: n for n in existing}
        if len(new_memories) != len(new_memories_meta):
            raise ValueError("each candidate requires metadata")
        meta = [
            dict(m, content=content, id=f"p{i}")
            for i, (content, m) in enumerate(zip(new_memories, new_memories_meta))
        ]
        ops: list[ReconcileOp] = []
        for start in range(0, len(meta), self.batch_size):
            proposals = meta[start : start + self.batch_size]
            questions: dict[str, dict] = {}
            if (
                start == 0
                and source_content
                and any(m.get("layer") != Layer.L3_DERIVED.value for m in meta)
            ):
                questions["coverage"] = {
                    "type": "choice",
                    "instructions": "Do the complete candidate proposals faithfully cover all explicit meaningful facts, updates, numbers and independent events in the incoming source? Do not demand inferred preferences or restated context. Check source text, not a coverage declaration. Missing explicit updates or unsupported invented candidates require INVALID.",
                    "criteria": {
                        "VALID": "Faithful explicit coverage",
                        "INVALID": "Missing or invented meaningful content",
                    },
                }
            for offset, item in enumerate(proposals):
                pid = item["id"]
                earlier = meta[: start + offset]
                target_ids = {n.node_id: n.content for n in existing}
                target_ids.update({p["id"]: p["content"] for p in earlier})
                if len(target_ids) > 253:
                    raise DecisionContextError(
                        "proposal target choices exceed Jev budget; split source into smaller writes"
                    )
                questions[f"{pid}:action"] = {
                    "type": "choice",
                    "instructions": f"Decide the executable action for proposals id={pid}. The source is incoming NEW evidence used only to verify support. The incoming source and stored user claims refer to the SAME scoped user. Explicit change/back/now in the source resolves the changed dimension; event records and state profiles are separate. Only current heads (or earlier proposals) represent stored knowledge; the proposal itself and matching source text NEVER make it already known. First verify source faithfulness (INVALID if invented or incorrectly attributed) and durable persistence value, then compare current heads, history and earlier proposals. Past occurrence is not current knowledge. Preserve unrelated dimensions and every distinct event. Do not infer truth from store time. If a current target contains multiple dimensions, choose NEED_CONTEXT unless a lossless merge preserves the unaffected dimensions.",
                    "criteria": {
                        key: description
                        for key, description in _ACTIONS.items()
                        if target_ids
                        or key
                        not in (
                            "SKIP",
                            "DELETE",
                            "ADD_OVERRIDE",
                            "ADD_MERGE",
                            "ADD_NEGATE",
                            "ADD_CONFLICT",
                        )
                    },
                }
                if item.get("layer") != Layer.L3_DERIVED.value:
                    questions[f"{pid}:representation"] = {
                        "type": "choice",
                        "instructions": f"Choose representation for candidate {pid} using source evidence and corresponding current heads. Current state/location, explicit preference or identity belongs to PROFILE even when phrased as a report. Preserve a corresponding dimension's representation across updates. A distinct past relocation event belongs to FACT. Explicit intentions are not inferred goals.",
                        "criteria": {
                            "FACT": "Explicit fact or distinct event",
                            "PROFILE": "State, identity or profile dimension",
                            "INTENTION": "Explicit planned action",
                        },
                    }
                if target_ids:
                    questions[f"{pid}:target"] = {
                        "type": "choice",
                        "instructions": f"For proposal id={pid}, select the ONE corresponding current target or earlier proposal; do not select an ancestor merely because its content matches. Match the same dimension: residence to residence; a distinct relocation event does not target an old relocation event just because both mention cities. Choose NONE for independent events. Choose MULTIPLE if more than one is required. Questions are evaluated independently.",
                        "criteria": {
                            **target_ids,
                            "NONE": "No corresponding existing target",
                            "MULTIPLE": "Requires multiple targets; needs expanded evaluation",
                        },
                    }
            state = {
                "source": source_content,
                "current_time": current_time,
                "current": current,
                "proposals": (
                    meta if "coverage" in questions else meta[: start + len(proposals)]
                ),
            }
            if plan_feedback:
                state["previous_plan_rejected"] = plan_feedback
            answers = await self.jev.evaluate(state=state, questions=questions)
            if "coverage" in questions and answers["coverage"].choice != "VALID":
                raise DecisionContextError("Jev rejected incomplete source coverage")
            for item in proposals:
                pid = item["id"]
                answer = answers[f"{pid}:action"]
                action = answer.choice
                if action not in _ACTIONS:
                    raise DecisionContextError(
                        "unsupported action returned by decision provider"
                    )
                if action in ("NEED_CONTEXT", "INVALID"):
                    raise DecisionContextError(f"Jev requested more context for {pid}")
                target_answer = answers.get(f"{pid}:target")
                target = target_answer.choice if target_answer else "NONE"
                target_ids = {n.node_id: n.content for n in existing}
                target_ids.update({p["id"]: p["content"] for p in meta[: int(pid[1:])]})
                if target not in (*target_ids, "NONE", "MULTIPLE"):
                    raise DecisionContextError(
                        "unknown target returned by decision provider"
                    )
                if target == "MULTIPLE":
                    include = {
                        f"{pid}:{nid}": {
                            "type": "choice",
                            "instructions": f"Does the selected action {action} for proposal {pid} require target {nid}?",
                            "criteria": {
                                "INCLUDE": "Required for the action",
                                "EXCLUDE": "Unrelated",
                                "UNSURE": "Insufficient evidence",
                            },
                        }
                        for nid in target_ids
                    }
                    selected = await self.jev.evaluate(state=state, questions=include)
                    if any(a.choice == "UNSURE" for a in selected.values()):
                        raise DecisionContextError("uncertain multiple targets")
                    targets = [
                        nid
                        for nid in target_ids
                        if selected[f"{pid}:{nid}"].choice == "INCLUDE"
                    ]
                else:
                    targets = [] if target == "NONE" else [str(target)]
                update_type = (
                    action.removeprefix("ADD_")
                    if action and action.startswith("ADD_")
                    else ""
                )
                needs_target = (
                    update_type in ("OVERRIDE", "NEGATE", "MERGE") or action == "DELETE"
                )
                if needs_target and not targets:
                    raise DecisionContextError(
                        f"{action} requires a corresponding target"
                    )
                if update_type == "CONFLICT" and not targets:
                    raise DecisionContextError(
                        "CONFLICT requires corresponding evidence"
                    )
                if action == "SKIP" and not targets:
                    raise DecisionContextError(
                        "SKIP requires a covering current target"
                    )
                references = targets if needs_target else []
                representation = answers.get(f"{pid}:representation")
                kind = item.get("kind", "fact")
                layer = item.get("layer", Layer.L1_FACT.value)
                if representation:
                    if representation.choice not in ("FACT", "PROFILE", "INTENTION"):
                        raise DecisionContextError("unsupported representation")
                    layer = (
                        Layer.L2_PROFILE.value
                        if representation.choice == "PROFILE"
                        else Layer.L1_FACT.value
                    )
                    kind = {
                        "FACT": "fact",
                        "PROFILE": "profile",
                        "INTENTION": "intention",
                    }[representation.choice]
                if (
                    (valid_from := _timestamp(item.get("valid_from"))) is not None
                    and (valid_until := _timestamp(item.get("valid_until"))) is not None
                    and valid_from > valid_until
                ):
                    raise DecisionContextError("invalid temporal range")
                valid_until = _timestamp(item.get("valid_until"))
                op = ReconcileOp(
                    op=(
                        "ADD"
                        if update_type
                        else ("SKIP" if action == "RAW_ONLY" else str(action))
                    ),
                    content=item["content"],
                    layer=layer,
                    kind=kind,
                    supersedes=references if update_type else [],
                    target_ids=targets,
                    memory_id=(
                        targets[0] if targets and action in ("DELETE", "SKIP") else None
                    ),
                    update_type=update_type,
                    negation=update_type == "NEGATE",
                    tags=list(item.get("tags") or []),
                    proposal_ids=[pid],
                    source_ids=list(item.get("source_ids") or []),
                    new_index=int(pid[1:]) + 1,
                    conflicts_with=targets if update_type == "CONFLICT" else [],
                    temporal_scope=item.get("temporal_scope"),
                    valid_from=valid_from,
                    valid_until=valid_until,
                    expected_revisions={
                        nid: by_id[nid].revision for nid in targets if nid in by_id
                    },
                    decision={
                        "action": answer.model_dump(),
                        "target": target_answer.model_dump() if target_answer else None,
                    },
                    reason=_ACTIONS[str(action)],
                    scope_epoch=scope_epoch,
                )
                ops.append(op)
        merges = [op for op in ops if op.update_type == "MERGE"]
        if merges:
            # One generation call for the whole selected merge set.
            ops_by_id = {pid: op for op in ops for pid in op.proposal_ids}

            def fragments(nid: str, path: set[str]) -> list[str]:
                if nid in path:
                    raise DecisionContextError("cyclic merge dependency")
                if nid in by_id:
                    return [by_id[nid].content]
                dependency = ops_by_id.get(nid)
                if dependency is None:
                    raise DecisionContextError("missing merge dependency")
                if dependency.op == "SKIP" and dependency.memory_id:
                    return fragments(dependency.memory_id, path | {nid})
                texts = [dependency.content or ""]
                if dependency.update_type == "MERGE":
                    for target in dependency.supersedes:
                        texts.extend(fragments(target, path | {nid}))
                return texts

            def sources(nid: str, path: set[str]) -> list[str]:
                if nid in path:
                    raise DecisionContextError("cyclic merge dependency")
                if nid in by_id:
                    return list((by_id[nid].custom or {}).get("source_ids") or [nid])
                dependency = ops_by_id[nid]
                if dependency.op == "SKIP" and dependency.memory_id:
                    return sources(dependency.memory_id, path | {nid})
                result = list(dependency.source_ids)
                if dependency.update_type == "MERGE":
                    for target in dependency.supersedes:
                        result.extend(sources(target, path | {nid}))
                return result

            material = [
                {
                    "proposal_id": op.proposal_ids[0],
                    "new": op.content,
                    "old": [
                        text for nid in op.supersedes for text in fragments(nid, set())
                    ],
                }
                for op in merges
            ]
            data = await self.llm.chat_json(
                system='Combine compatible memory fragments without judging relations. Preserve every number, event and unrelated dimension. Return {"merges":[{"proposal_id":...,"content":...}]}. Never omit facts.',
                user=json.dumps(material, ensure_ascii=False),
                max_tokens=4096,
            )
            generated = {m["proposal_id"]: m["content"] for m in data.get("merges", [])}
            if any(
                not isinstance(text, str) or not text.strip()
                for text in generated.values()
            ):
                raise DecisionContextError("merge generation returned empty content")
            if set(generated) != {op.proposal_ids[0] for op in merges}:
                raise DecisionContextError("merge generation lacks source coverage")
            verify = {
                op.proposal_ids[0]: {
                    "type": "choice",
                    "instructions": f"Does merged text for {op.proposal_ids[0]} preserve all meaningful facts, all numbers and all distinct events from sources?",
                    "criteria": {
                        "VALID": "Lossless and faithful",
                        "INVALID": "Missing or invented detail",
                    },
                }
                for op in merges
            }
            checked = await self.jev.evaluate(
                state={"sources": material, "merged": generated},
                questions=verify,
            )
            if any(a.choice != "VALID" for a in checked.values()):
                raise DecisionContextError("Jev rejected lossy merge")
            for op in merges:
                op.content = generated[op.proposal_ids[0]]
                op.source_ids = list(
                    dict.fromkeys(
                        [
                            *op.source_ids,
                            *(
                                sid
                                for target in op.supersedes
                                for sid in sources(target, set())
                            ),
                        ]
                    )
                )
        return ops
