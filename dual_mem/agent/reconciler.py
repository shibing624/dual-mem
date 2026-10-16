"""Jev relation decisions over a bounded set of comparison material.

Embedding similarity only *orders* which current heads Jev compares against when
the scope exceeds the material or context budget. It never decides whether a
candidate is written: every proposal still receives a Jev decision.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from dual_mem.errors import CoverageError, DecisionContextError, DecisionPlanError
from dual_mem.isolation import build_filter
from dual_mem.providers.jev import DecisionProvider, JevAnswer
from dual_mem.providers.llm import LLMClient
from dual_mem.storage.vector_store import VectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.agent.reconciler")

_ACTIONS = {
    "ADD_OVERRIDE": "Replace a corresponding PERSISTENT STATE VALUE or explicitly correct a prior claim. Current residence replaces previous residence, even when returning to an ancestor value. NEVER replace a different completed event: two moves are two retained events.",
    "ADD_MERGE": "Complementary fragments about ONE object; preserve every number, source and distinct event.",
    "ADD_SUPPLEMENT": "Append a new dimension or a DISTINCT OCCURRENCE. A genuinely different completed move/purchase/visit is an independent event, even when it changes a state or repeats an earlier value. Repeating a report of the SAME occurrence is SKIP, not a new event. Keep all earlier events. Initial state without baseline also belongs here.",
    "ADD_TEMPORAL": "Explicit LIMITED exception (this week/temporarily/until a date) coexisting with a long-term baseline. 'Currently/目前/现在' alone is not a limited exception.",
    "ADD_NEGATE": "Explicit termination or denial of an existing claim.",
    "ADD_CONFLICT": "Competing SIMULTANEOUS incompatible claims with no explicit change, correction, ordering or authority. A user's explicit move/change/back is OVERRIDE, not CONFLICT. Different sequential events are SUPPLEMENT, not CONFLICT.",
    "ADD_HISTORICAL": "The incoming claim is an OLDER report of the target's state: its memory_at precedes the target's. Record it as a past version; the newer target stays current.",
    "RAW_ONLY": "No durable fact, event, profile or intention; retain source evidence only.",
    "SKIP": "A corresponding current head or earlier proposal ALREADY FULLY COVERS this exact claim, including any reported change/event and time scope. Repeating “moved back/again” does not create a new move when it reports the SAME occurrence and current state. Prefer SKIP over OVERRIDE when all content and state are already current. Ancestors and the incoming source alone cannot establish knownness.",
    "DELETE": "Explicit retraction or redundant representation with a retained canonical source; archive, not erase.",
    "INVALID": "Candidate is unsupported by source, invented or incorrectly attributed; reject extraction.",
    "NEED_CONTEXT": "Evidence is insufficient or inconsistent; never guess based on insertion time.",
}

_WITH_TARGET = (
    "ADD_OVERRIDE",
    "ADD_MERGE",
    "ADD_NEGATE",
    "ADD_CONFLICT",
    "SKIP",
    "DELETE",
    "INVALID",
    "NEED_CONTEXT",
)
# A source older than its target may not retire that target: OVERRIDE, NEGATE,
# DELETE and MERGE all end the head's currency, so none is offered.
_WITH_NEWER_TARGET = (
    "ADD_HISTORICAL",
    "ADD_CONFLICT",
    "SKIP",
    "INVALID",
    "NEED_CONTEXT",
)
_WITHOUT_TARGET = (
    "ADD_SUPPLEMENT",
    "ADD_TEMPORAL",
    "RAW_ONLY",
    "INVALID",
    "NEED_CONTEXT",
)
_TARGET_REQUIRED_ACTIONS: set[str] = {
    *_WITH_TARGET,
    *_WITH_NEWER_TARGET,
} - {"INVALID", "NEED_CONTEXT"}
_OPTION_CHARS = 240
# Instructions of the target/action/coverage questions plus JSON framing.
_QUESTION_OVERHEAD = 4000


def _action_question(
    pid: str,
    *,
    has_targets: bool,
    allow_delete: bool = True,
    allow_skip: bool = True,
    older_than_target: bool = False,
) -> dict[str, Any]:
    if has_targets:
        skip_instruction = (
            "SKIP when the single target already fully covers this claim, including the reported event and state. "
            if allow_skip
            else (
                "Multiple targets cannot be represented by one SKIP alias. "
                if older_than_target
                else "Multiple targets cannot be represented by one SKIP alias; use MERGE when they jointly cover the claim. "
            )
        )
        if older_than_target:
            replace_instruction = (
                "The source is OLDER than a selected target (compare memory_at), so it cannot replace, merge into, negate or delete it; the target stays current. "
                "ADD_HISTORICAL when it reports an earlier value or an earlier fragment of that same object. "
            )
            delete_instruction = ""
            merge_instruction = ""
        else:
            replace_instruction = (
                "ADD_OVERRIDE when this proposal replaces that same persistent state. "
                "ADD_NEGATE when the source terminates the target. "
            )
            delete_instruction = (
                "DELETE when the source retracts the current target. "
                if allow_delete
                else "An earlier proposal cannot be deleted before it is committed. "
            )
            merge_instruction = "ADD_MERGE when both are fragments of one object and every unaffected fact survives. "
        instructions = (
            f"Decide the relation for proposal id={pid}. "
            "state.selected_targets already names one or more current heads or earlier proposals. "
            "Use only those targets. "
            f"{skip_instruction}"
            f"{replace_instruction}"
            f"{merge_instruction}"
            "ADD_CONFLICT when both claims stay current and incompatible. "
            f"{delete_instruction}"
            "A completed different event is not a relation to this target. "
            "INVALID if the candidate is invented or misattributed. "
            "NEED_CONTEXT if the target mixes dimensions and a lossless merge is impossible. "
            "The incoming source text itself is not stored knowledge."
        )
        base = _WITH_NEWER_TARGET if older_than_target else _WITH_TARGET
        keys = tuple(
            key
            for key in base
            if (allow_delete or key != "DELETE") and (allow_skip or key != "SKIP")
        )
    else:
        instructions = (
            f"Decide the action for proposal id={pid}. "
            "No current head or earlier proposal was selected, so there is no version to update, cover, negate, conflict with, or delete. "
            "ADD_SUPPLEMENT appends a new dimension or a distinct occurrence, including an initial state with no baseline. "
            "ADD_TEMPORAL is only an explicit limited exception and does not link a baseline. "
            "RAW_ONLY keeps the source when nothing durable should be stored. "
            "INVALID if the candidate is invented or misattributed. "
            "NEED_CONTEXT if the source is insufficient. "
            "Do not change, cover, or delete an old version; no such target was selected."
        )
        keys = _WITHOUT_TARGET
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {key: _ACTIONS[key] for key in keys},
    }


@dataclass
class ReconcileOp:
    """One executable decision. ``op`` is ADD / SKIP / DELETE / RAW_ONLY / INVALID."""

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
    proposal_ids: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    expected_revisions: dict[str, int] = field(default_factory=dict)
    source_ids: list[str] = field(default_factory=list)
    valid_from: int | None = None
    valid_until: int | None = None
    decision: dict[str, Any] = field(default_factory=dict)
    scope_epoch: int = 0
    target_ids: list[str] = field(default_factory=list)
    # Written as an independent SUPPLEMENT until review resolves its relation.
    needs_review: bool = False
    review_reason: str = ""
    review_targets: list[str] = field(default_factory=list)
    # Proposal embedding reused at write time when the content is unchanged.
    embedding: list[float] | None = None


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
            "needs_review": op.needs_review,
            "review_reason": op.review_reason,
            "review_targets": op.review_targets,
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


def _date(ts: int | None) -> str:
    if not ts:
        return "undated"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


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
        max_state_chars: int = 60_000,
        review_confidence: float = 0.0,
    ) -> None:
        self.jev, self.llm, self.embed, self.vector = jev, llm, embed, vector
        self.candidate_limit, self.batch_size = candidate_limit, batch_size
        self.max_state_chars = max_state_chars
        self.review_confidence = review_confidence

    async def _evaluate(
        self,
        *,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, JevAnswer]:
        try:
            answers = await self.jev.evaluate(state=state, questions=questions)
        except ValueError as exc:
            raise DecisionContextError("Jev request or response rejected") from exc
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise DecisionContextError("Jev response lacks exact question coverage")
        checked: dict[str, JevAnswer] = {}
        for name, raw in answers.items():
            try:
                answer = JevAnswer.model_validate(raw)
            except (TypeError, ValueError) as exc:
                raise DecisionContextError("invalid Jev answer") from exc
            question = questions[name]
            if answer.type != question["type"]:
                raise DecisionContextError("Jev answer type does not match question")
            if answer.type == "choice" and answer.choice not in question["criteria"]:
                raise DecisionContextError("Jev selected an unsupported answer")
            checked[name] = answer
        return checked

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

    @staticmethod
    def _option(item: dict) -> str:
        """Short choice label; the full node is already in ``state.current``."""
        text = str(item.get("content") or "")
        if len(text) > _OPTION_CHARS:
            text = text[:_OPTION_CHARS] + "…"
        kind = item.get("kind") or "fact"
        return f"[{item.get('layer')}/{kind} {_date(item.get('memory_at'))}] {text}"

    async def _route(
        self,
        *,
        where: dict,
        meta: list[dict],
        excluded: set[str],
        embeddings: dict[str, list[float]],
    ) -> list[MemoryNode]:
        """Order current heads by their best similarity to any proposal."""
        missing = [m for m in meta if m["id"] not in embeddings]
        if missing:
            vectors = await self.embed.embed_batch([m["content"] for m in missing])
            embeddings.update({m["id"]: v for m, v in zip(missing, vectors)})
        best: dict[str, MemoryNode] = {}
        for item in meta:
            for node in self.vector.query(
                embedding=embeddings[item["id"]],
                where=where,
                top_k=self.candidate_limit + len(excluded),
            ):
                if node.node_id in excluded:
                    continue
                if node.node_id not in best or node.score > best[node.node_id].score:
                    best[node.node_id] = node
        ranked = sorted(best.values(), key=lambda n: n.score, reverse=True)
        return ranked[: self.candidate_limit]

    async def _material(
        self,
        *,
        scope: tuple[str, str, str],
        meta: list[dict],
        excluded: set[str],
        base_chars: int,
        embeddings: dict[str, list[float]],
    ) -> list[tuple[MemoryNode, dict]]:
        where = build_filter(
            app_ids=[scope[0]],
            user_id=scope[1],
            agent_ids=[scope[2]],
            layers=[Layer.L1_FACT, Layer.L2_PROFILE, Layer.L3_DERIVED],
            latest_only=True,
            statuses=[MemoryStatus.ACTIVE],
        )
        heads = [
            n
            for n in self.vector.get_many(
                where, limit=self.candidate_limit + len(excluded) + 1
            )
            if n.node_id not in excluded
        ]
        ranked = len(heads) > self.candidate_limit
        if ranked:
            heads = await self._route(
                where=where, meta=meta, excluded=excluded, embeddings=embeddings
            )

        def contexts(nodes: list[MemoryNode]) -> list[tuple[MemoryNode, dict]]:
            return [
                (n, dict(self._context_node(n), history_versions=self.history(n, scope)))
                for n in nodes
            ]

        def size(item: dict) -> int:
            return len(json.dumps(item, ensure_ascii=False)) + _OPTION_CHARS + 64

        material = contexts(heads)
        sizes = [size(c) for _, c in material]
        if base_chars + sum(sizes) > self.max_state_chars and not ranked:
            material = contexts(
                await self._route(
                    where=where, meta=meta, excluded=excluded, embeddings=embeddings
                )
            )
            sizes = [size(c) for _, c in material]
        dropped = 0
        while material and base_chars + sum(sizes) > self.max_state_chars:
            material.pop()
            sizes.pop()
            dropped += 1
        if dropped:
            logger.warning(
                "reconcile scope=%s: %d lowest-ranked heads omitted to fit the Jev budget",
                scope,
                dropped,
            )
        return material

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
        memory_at: int | None = None,
        check_coverage: bool = True,
    ) -> list[ReconcileOp]:
        if not new_memories:
            return []
        if len(new_memories) != len(new_memories_meta):
            raise ValueError("each candidate requires metadata")
        scope = (app_id, user_id, agent_id)
        scope_epoch = self.vector.scope_epoch(scope)
        meta = [
            dict(m, content=content, id=f"p{i}")
            for i, (content, m) in enumerate(zip(new_memories, new_memories_meta))
        ]
        base_state: dict[str, Any] = {
            "source": source_content,
            "current_time": current_time,
            "proposals": meta,
        }
        if plan_feedback:
            base_state["previous_plan_rejected"] = plan_feedback
        base_chars = (
            len(json.dumps(base_state, ensure_ascii=False))
            + len(meta) * (_OPTION_CHARS + 64)
            + _QUESTION_OVERHEAD
        )
        if base_chars > self.max_state_chars:
            raise DecisionContextError(
                "source and proposals exceed the Jev context budget; split the source"
            )
        embeddings: dict[str, list[float]] = {}
        material = await self._material(
            scope=scope,
            meta=meta,
            excluded=set(exclude_ids or []),
            base_chars=base_chars,
            embeddings=embeddings,
        )
        existing = [n for n, _ in material]
        current = [c for _, c in material]
        by_id = {n.node_id: n for n in existing}
        head_options = {n.node_id: self._option(c) for n, c in zip(existing, current)}
        proposal_ids = {item["id"] for item in meta}
        ops: list[ReconcileOp] = []
        for start in range(0, len(meta), self.batch_size):
            proposals = meta[start : start + self.batch_size]
            questions: dict[str, dict] = {}
            target_options: dict[str, dict[str, str]] = {}
            if (
                check_coverage
                and start == 0
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
                options = dict(head_options)
                options.update({p["id"]: self._option(p) for p in earlier})
                if len(options) > 253:
                    raise DecisionContextError(
                        "proposal target choices exceed Jev budget; split source into smaller writes"
                    )
                target_options[pid] = options
                if not options:
                    questions[f"{pid}:action"] = _action_question(
                        pid, has_targets=False
                    )
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
                if options:
                    questions[f"{pid}:target"] = {
                        "type": "choice",
                        "instructions": f"For proposal id={pid}, select the ONE executable relation target before deciding its action. Option labels are short; the full nodes and their history_versions are in state.current. Select a current head or earlier proposal only when this proposal covers, replaces, merges, negates or conflicts with it, or is an older report of that same state. Choose NONE for an independent supplement, an explicit limited TEMPORAL exception, or RAW_ONLY, even when a baseline is relevant context. An explicit retraction must select the persisted current head directly because an earlier proposal cannot be deleted. Do not select an ancestor merely because its content matches. Match the SAME representation and dimension: a residence state must target a residence state, never a move event. A completed event targets NONE unless explicitly correcting that SAME occurrence. A-to-B then B-to-A are different events; choose NONE. For repeated residence, select the residence profile that covers the claim, not a move mentioning that city. Choose MULTIPLE if more than one is required.",
                        "criteria": {
                            **options,
                            "NONE": "No corresponding existing target",
                            "MULTIPLE": "Requires multiple targets; needs expanded evaluation",
                        },
                    }
            state = dict(
                base_state,
                current=current,
                proposals=(
                    meta if "coverage" in questions else meta[: start + len(proposals)]
                ),
            )
            answers = await self._evaluate(state=state, questions=questions)
            if "coverage" in questions and answers["coverage"].choice != "VALID":
                raise CoverageError("Jev rejected incomplete source coverage")
            targets_by_pid: dict[str, list[str]] = {}
            uncertain: dict[str, list[str]] = {}
            for item in proposals:
                pid = item["id"]
                target_answer = answers.get(f"{pid}:target")
                target = target_answer.choice if target_answer else "NONE"
                options = target_options[pid]
                if target not in (*options, "NONE", "MULTIPLE"):
                    raise DecisionContextError(
                        "unknown target returned by decision provider"
                    )
                if target == "MULTIPLE":
                    include = {
                        f"{pid}:{nid}": {
                            "type": "choice",
                            "instructions": f"Is target {nid} required as corresponding evidence for proposal {pid}?",
                            "criteria": {
                                "INCLUDE": "Required corresponding evidence",
                                "EXCLUDE": "Unrelated",
                                "UNSURE": "Insufficient evidence",
                            },
                        }
                        for nid in options
                    }
                    selected = await self._evaluate(state=state, questions=include)
                    targets = [
                        nid
                        for nid in options
                        if selected[f"{pid}:{nid}"].choice == "INCLUDE"
                    ]
                    if any(a.choice == "UNSURE" for a in selected.values()):
                        # Relation is unresolved: append independently, review later.
                        uncertain[pid] = [
                            nid
                            for nid in options
                            if selected[f"{pid}:{nid}"].choice != "EXCLUDE"
                            and nid not in proposal_ids
                        ]
                        targets = []
                else:
                    targets = [] if target == "NONE" else [str(target)]
                targets_by_pid[pid] = targets

            def older(targets: list[str]) -> bool:
                return memory_at is not None and any(
                    by_id[t].memory_at is not None and by_id[t].memory_at > memory_at
                    for t in targets
                    if t in by_id
                )

            action_questions = {
                f"{item['id']}:action": _action_question(
                    item["id"],
                    has_targets=bool(targets_by_pid[item["id"]]),
                    allow_delete=not any(
                        target in proposal_ids for target in targets_by_pid[item["id"]]
                    ),
                    allow_skip=len(targets_by_pid[item["id"]]) == 1,
                    older_than_target=older(targets_by_pid[item["id"]]),
                )
                for item in proposals
                if f"{item['id']}:action" not in answers
            }
            if action_questions:
                answers.update(
                    await self._evaluate(
                        state={**state, "selected_targets": targets_by_pid},
                        questions=action_questions,
                    )
                )

            for item in proposals:
                ops.append(
                    self._build_op(
                        item=item,
                        answers=answers,
                        targets=targets_by_pid[item["id"]],
                        uncertain=uncertain.get(item["id"]),
                        proposal_ids=proposal_ids,
                        by_id=by_id,
                        scope_epoch=scope_epoch,
                    )
                )
        await self._generate_merges(ops, by_id)
        for op in ops:
            if op.op == "ADD" and op.update_type != "MERGE":
                op.embedding = embeddings.get(op.proposal_ids[0])
        return ops

    def _build_op(
        self,
        *,
        item: dict,
        answers: dict[str, JevAnswer],
        targets: list[str],
        uncertain: list[str] | None,
        proposal_ids: set[str],
        by_id: dict[str, MemoryNode],
        scope_epoch: int,
    ) -> ReconcileOp:
        pid = item["id"]
        answer = answers[f"{pid}:action"]
        target_answer = answers.get(f"{pid}:target")
        action = str(answer.choice)
        if action not in _ACTIONS:
            raise DecisionContextError("unsupported action returned by decision provider")
        relation = action in _TARGET_REQUIRED_ACTIONS
        if action != "NEED_CONTEXT" and action != "INVALID" and relation != bool(targets):
            raise DecisionContextError(f"{action} does not match the selected target")
        if action == "DELETE" and any(t in proposal_ids for t in targets):
            raise DecisionContextError("DELETE cannot target an uncommitted proposal")
        if action == "SKIP" and len(targets) != 1:
            raise DecisionContextError("SKIP requires exactly one covering target")

        review_reason = ""
        review_targets: list[str] = []
        effective = action
        if uncertain is not None and action not in ("RAW_ONLY", "INVALID"):
            review_reason, review_targets = "uncertain multiple targets", uncertain
            if action == "NEED_CONTEXT":
                effective = "ADD_SUPPLEMENT"
        elif action == "NEED_CONTEXT":
            review_reason, review_targets = "NEED_CONTEXT", list(targets)
            effective = "ADD_SUPPLEMENT"
        elif relation and (answer.confidence or 0.0) < self.review_confidence:
            review_reason = f"low-confidence {action}"
            review_targets = list(targets)
            effective = "ADD_SUPPLEMENT"
        review_targets = [t for t in review_targets if t not in proposal_ids]
        linked = targets if effective in _TARGET_REQUIRED_ACTIONS else []
        update_type = effective.removeprefix("ADD_") if effective.startswith("ADD_") else ""
        if update_type == "HISTORICAL" and any(t not in by_id for t in linked):
            raise DecisionPlanError("HISTORICAL must attach to a persisted current head")

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
            kind = {"FACT": "fact", "PROFILE": "profile", "INTENTION": "intention"}[
                representation.choice
            ]
        valid_from = _timestamp(item.get("valid_from"))
        valid_until = _timestamp(item.get("valid_until"))
        if valid_from is not None and valid_until is not None and valid_from > valid_until:
            raise DecisionContextError("invalid temporal range")
        decision: dict[str, Any] = {
            "action": answer.model_dump(),
            "target": target_answer.model_dump() if target_answer else None,
        }
        if effective != action:
            decision["effective"] = effective
        return ReconcileOp(
            op="ADD" if update_type else effective,
            content=item["content"],
            layer=layer,
            kind=kind,
            supersedes=(linked if update_type in ("OVERRIDE", "NEGATE", "MERGE") else []),
            target_ids=linked,
            memory_id=linked[0] if linked and effective in ("DELETE", "SKIP") else None,
            update_type=update_type,
            negation=update_type == "NEGATE",
            tags=list(item.get("tags") or []),
            proposal_ids=[pid],
            source_ids=list(item.get("source_ids") or []),
            conflicts_with=linked if update_type == "CONFLICT" else [],
            temporal_scope=item.get("temporal_scope"),
            valid_from=valid_from,
            valid_until=valid_until,
            expected_revisions={nid: by_id[nid].revision for nid in linked if nid in by_id},
            decision=decision,
            reason=_ACTIONS[action],
            scope_epoch=scope_epoch,
            needs_review=bool(review_reason),
            review_reason=review_reason,
            review_targets=review_targets,
        )

    async def _generate_merges(
        self, ops: list[ReconcileOp], by_id: dict[str, MemoryNode]
    ) -> None:
        merges = [op for op in ops if op.update_type == "MERGE"]
        if not merges:
            return
        # One generation call for the whole selected merge set.
        ops_by_id = {pid: op for op in ops for pid in op.proposal_ids}

        def dependency_of(nid: str, path: set[str]) -> ReconcileOp:
            if nid in path:
                raise DecisionContextError("cyclic merge dependency")
            dependency = ops_by_id.get(nid)
            if dependency is None:
                raise DecisionContextError("missing merge dependency")
            if dependency.op != "ADD" and not (
                dependency.op == "SKIP" and dependency.memory_id
            ):
                raise DecisionPlanError(
                    "operation targets an earlier proposal that produced no memory"
                )
            return dependency

        def fragments(nid: str, path: set[str]) -> list[str]:
            if nid in by_id:
                return [by_id[nid].content]
            dependency = dependency_of(nid, path)
            if dependency.op == "SKIP":
                return fragments(str(dependency.memory_id), path | {nid})
            texts = [dependency.content or ""]
            if dependency.update_type == "MERGE":
                for target in dependency.supersedes:
                    texts.extend(fragments(target, path | {nid}))
            return texts

        def sources(nid: str, path: set[str]) -> list[str]:
            if nid in by_id:
                return list((by_id[nid].custom or {}).get("source_ids") or [nid])
            dependency = dependency_of(nid, path)
            if dependency.op == "SKIP":
                return sources(str(dependency.memory_id), path | {nid})
            result = list(dependency.source_ids)
            if dependency.update_type == "MERGE":
                for target in dependency.supersedes:
                    result.extend(sources(target, path | {nid}))
            return result

        material = [
            {
                "proposal_id": op.proposal_ids[0],
                "new": op.content,
                "old": [text for nid in op.supersedes for text in fragments(nid, set())],
            }
            for op in merges
        ]
        data = await self.llm.chat_json(
            system='Combine compatible memory fragments without judging relations. Preserve every number, event and unrelated dimension. Return {"merges":[{"proposal_id":...,"content":...}]}. Never omit facts.',
            user=json.dumps(material, ensure_ascii=False),
            max_tokens=4096,
        )
        generated = {m["proposal_id"]: m["content"] for m in data.get("merges", [])}
        if any(not isinstance(text, str) or not text.strip() for text in generated.values()):
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
        checked = await self._evaluate(
            state={"sources": material, "merged": generated}, questions=verify
        )
        if any(a.choice != "VALID" for a in checked.values()):
            raise DecisionContextError("Jev rejected lossy merge")
        for op in merges:
            op.content = generated[op.proposal_ids[0]]
            op.source_ids = list(
                dict.fromkeys(
                    [
                        *op.source_ids,
                        *(sid for target in op.supersedes for sid in sources(target, set())),
                    ]
                )
            )


__all__ = [
    "CoverageError",
    "DecisionContextError",
    "DecisionPlanError",
    "ReconcileOp",
    "Reconciler",
    "build_apply_custom",
    "collect_merged_timestamps",
    "reflect_custom",
]
