"""Jev relation decisions over a bounded set of comparison material.

Embedding similarity bounds which current heads Jev compares against. It never
discards a candidate: without a comparison target the candidate is appended;
otherwise Jev chooses its relation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from dual_mem.errors import CoverageError, DecisionContextError, DecisionPlanError
from dual_mem.isolation import build_filter
from dual_mem.providers.jev import DecisionProvider, JevAnswer
from dual_mem.providers.llm import LLMClient
from dual_mem.storage.vector_store import VectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.agent.reconciler")

_ACTIONS = {
    "ADD_OVERRIDE": "Replace a corresponding PERSISTENT STATE VALUE or explicitly correct a prior claim. Current residence replaces previous residence, even when returning to an ancestor value. NEVER replace a different completed event: two moves are two retained events.",
    "ADD_SUPPLEMENT": "Append a new dimension or a DISTINCT OCCURRENCE. A genuinely different completed move/purchase/visit is an independent event, even when it changes a state or repeats an earlier value. Repeating a report of the SAME occurrence is SKIP, not a new event. Keep all earlier events. Initial state without baseline also belongs here.",
    "ADD_TEMPORAL": "Explicit LIMITED exception (this week/temporarily/until a date) coexisting with a long-term baseline. 'Currently/目前/现在' alone is not a limited exception.",
    "ADD_NEGATE": "Explicit termination or denial of an existing claim, including the user withdrawing a detail they asked the assistant to forget or stop using: the withdrawal replaces the detail.",
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
    "ADD_SUPPLEMENT",
    "ADD_NEGATE",
    "ADD_CONFLICT",
    "SKIP",
    "DELETE",
    "INVALID",
    "NEED_CONTEXT",
)
# A source older than its target may not retire that target: OVERRIDE, NEGATE,
# DELETE ends the head's currency too, so it is not offered here.
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
} - {"INVALID", "NEED_CONTEXT", "ADD_SUPPLEMENT"}
_OPTION_CHARS = 240
# Older versions shown per candidate head; the newest ones decide currency.
_HISTORY_LIMIT = 3
_TARGET_INSTRUCTIONS = (
    "For proposal id={pid}, select the ONE executable relation target. Option labels are short; "
    "the full nodes and their history_versions are in state.current (earlier proposals in state.proposals). "
    "Select a target only when this proposal repeats, replaces, negates, retracts or conflicts with it, "
    "or is an older report of that same state. Match the SAME dimension: a residence state targets a "
    "residence state, never a move event. A distinct completed event, a new dimension, an explicit "
    "limited exception or a complementary detail is NONE, even when the topic is related."
)
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
                else "Multiple targets cannot be represented by one SKIP alias; use ADD_SUPPLEMENT when they jointly cover the claim. "
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
            # No ADD_MERGE: combining texts costs a generation plus a losslessness
            # check per batch and fails on any omitted detail. Complementary fragments
            # are written as separate memories and read as a group (same object).
            merge_instruction = "Use ADD_SUPPLEMENT when both are complementary fragments of one object; both are kept."
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
    owner: str = ""
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


def reflect_custom(op: ReconcileOp) -> dict:
    return {
        key: value
        for key, value in {
            "update_type": op.update_type,
            "temporal_scope": op.temporal_scope,
            "negation": op.negation,
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
    custom = reflect_custom(op)
    if source_node_id:
        custom["source_node_id"] = source_node_id
    return custom


def _date(ts: int | None) -> str:
    if not ts:
        return "undated"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


class Reconciler:
    """Batch atomic Jev questions, then execute the resulting relations."""

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
        target_limit: int = 5,
        relation_threshold: float = 0.6,
    ) -> None:
        self.jev, self.llm, self.embed, self.vector = jev, llm, embed, vector
        self.target_limit = target_limit
        self.relation_threshold = relation_threshold
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
        material = {
            "id": node.node_id,
            "content": node.content,
            "layer": node.layer.value,
            "kind": node.kind,
            "is_latest": node.is_latest,
            "memory_at": node.memory_at,
            "valid_from": node.valid_from,
            "valid_until": node.valid_until,
            "update_type": (node.custom or {}).get("update_type"),
            "owner": node.owner,
            "conflicts_with": (node.custom or {}).get("conflicts_with", []),
        }
        return {key: value for key, value in material.items() if value not in (None, "", [])}

    @staticmethod
    def _option(item: dict) -> str:
        """Short choice label; the full node is already in ``state.current``."""
        text = str(item.get("content") or "")
        if len(text) > _OPTION_CHARS:
            text = text[:_OPTION_CHARS] + "…"
        kind = item.get("kind") or "fact"
        return f"[{item.get('layer')}/{kind} {_date(item.get('memory_at'))}] {text}"

    async def _neighbours(
        self,
        *,
        where: dict,
        meta: list[dict],
        excluded: set[str],
        embeddings: dict[str, list[float]],
    ) -> dict[str, list[MemoryNode]]:
        """Per proposal: the few current heads similar enough to be a relation target."""
        missing = [m for m in meta if m["id"] not in embeddings]
        if missing:
            vectors = await self.embed.embed_batch([m["content"] for m in missing])
            embeddings.update({m["id"]: v for m, v in zip(missing, vectors)})
        out: dict[str, list[MemoryNode]] = {}
        for item in meta:
            nodes = self.vector.query(
                embedding=embeddings[item["id"]],
                where=where,
                top_k=self.target_limit + len(excluded),
            )
            out[item["id"]] = [
                n
                for n in nodes
                if n.node_id not in excluded and n.score >= self.relation_threshold
            ][: self.target_limit]
        return out

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
        check_coverage: bool = False,
    ) -> list[ReconcileOp]:
        """Decide every proposal with at most two Jev rounds over a small state.

        - A proposal with no current head or earlier proposal above
          ``relation_threshold`` cosine has nothing to update, cover or conflict with:
          it is appended directly (SUPPLEMENT, or TEMPORAL when it carries an explicit
          validity bound). No model call is spent on it.
        - Every other proposal gets ONE target question whose options are only its own
          nearest heads/earlier proposals (<= ``target_limit``), then one action
          question when a target was chosen. Choosing NONE appends directly.
        - The Jev state holds proposals and the union of candidate heads only; the
          source conversation is not resent (candidates are already faithful text).
        """
        del source_content, plan_feedback, check_coverage
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
        proposal_ids = {item["id"] for item in meta}
        base_chars = (
            len(json.dumps({"current_time": current_time, "proposals": meta}, ensure_ascii=False))
            + _QUESTION_OVERHEAD
        )
        if base_chars > self.max_state_chars:
            raise DecisionContextError(
                "source and proposals exceed the Jev context budget; split the source"
            )
        where = build_filter(
            app_ids=[scope[0]],
            user_id=scope[1],
            agent_ids=[scope[2]],
            layers=[Layer.L1_FACT, Layer.L2_PROFILE, Layer.L3_DERIVED],
            latest_only=True,
            statuses=[MemoryStatus.ACTIVE],
        )
        embeddings: dict[str, list[float]] = {}
        near = await self._neighbours(
            where=where, meta=meta, excluded=set(exclude_ids or []), embeddings=embeddings
        )
        by_id: dict[str, MemoryNode] = {n.node_id: n for ns in near.values() for n in ns}

        # Earlier proposals of the same write are relation targets only when similar.
        vec = {pid: np.asarray(v, dtype=float) for pid, v in embeddings.items()}
        for pid in vec:
            norm = float(np.linalg.norm(vec[pid]))
            if norm:
                vec[pid] = vec[pid] / norm
        options: dict[str, dict[str, str]] = {}
        contexts: dict[str, dict] = {}
        for idx, item in enumerate(meta):
            pid = item["id"]
            opts: dict[str, str] = {}
            for node in near[pid]:
                if node.node_id not in contexts:
                    contexts[node.node_id] = dict(
                        self._context_node(node),
                        history_versions=self.history(node, scope)[:_HISTORY_LIMIT],
                    )
                opts[node.node_id] = self._option(contexts[node.node_id])
            earlier = sorted(
                (
                    (float(vec[pid] @ vec[p["id"]]), p)
                    for p in meta[:idx]
                ),
                key=lambda x: -x[0],
            )
            for sim, p in earlier[: self.target_limit]:
                if sim >= self.relation_threshold:
                    opts[p["id"]] = self._option(p)
            options[pid] = opts

        ops: list[ReconcileOp | None] = [None] * len(meta)
        undecided: list[int] = []
        for idx, item in enumerate(meta):
            if options[item["id"]]:
                undecided.append(idx)
            else:
                ops[idx] = self._routed_op(item, proposal_ids, by_id, scope_epoch)

        async def decide(chunk: list[int]) -> None:
            proposals = [meta[i] for i in chunk]
            head_ids = {
                nid for item in proposals for nid in options[item["id"]] if nid in contexts
            }
            refs = {
                nid for item in proposals for nid in options[item["id"]] if nid in proposal_ids
            }
            state: dict[str, Any] = {
                "current_time": current_time,
                "proposals": [p for p in meta if p["id"] in refs or p in proposals],
                "current": [contexts[nid] for nid in contexts if nid in head_ids],
            }
            # Keep the state inside the Jev budget by dropping the least similar heads.
            while (
                state["current"]
                and len(json.dumps(state, ensure_ascii=False)) + _QUESTION_OVERHEAD
                > self.max_state_chars
            ):
                dropped = state["current"].pop()["id"]
                for item in proposals:
                    options[item["id"]].pop(dropped, None)
            questions = {
                f"{item['id']}:target": {
                    "type": "choice",
                    "instructions": _TARGET_INSTRUCTIONS.format(pid=item["id"]),
                    "criteria": {
                        **options[item["id"]],
                        "NONE": "No corresponding existing target",
                    },
                }
                for item in proposals
            }
            answers = await self._evaluate(state=state, questions=questions)
            targets_by_pid: dict[str, list[str]] = {}
            for item in proposals:
                choice = answers[f"{item['id']}:target"].choice
                if choice not in (*options[item["id"]], "NONE"):
                    raise DecisionContextError("unknown target returned by decision provider")
                targets_by_pid[item["id"]] = [] if choice == "NONE" else [str(choice)]

            def older(targets: list[str]) -> bool:
                return memory_at is not None and any(
                    by_id[t].memory_at is not None and by_id[t].memory_at > memory_at
                    for t in targets
                    if t in by_id
                )

            action_questions = {
                f"{item['id']}:action": _action_question(
                    item["id"],
                    has_targets=True,
                    allow_delete=not any(
                        t in proposal_ids for t in targets_by_pid[item["id"]]
                    ),
                    allow_skip=True,
                    older_than_target=older(targets_by_pid[item["id"]]),
                )
                for item in proposals
                if targets_by_pid[item["id"]]
            }
            if action_questions:
                answers.update(
                    await self._evaluate(
                        state={**state, "selected_targets": {
                            k: v for k, v in targets_by_pid.items() if v
                        }},
                        questions=action_questions,
                    )
                )
            for i, item in zip(chunk, proposals):
                if not targets_by_pid[item["id"]]:
                    ops[i] = self._routed_op(
                        item,
                        proposal_ids,
                        by_id,
                        scope_epoch,
                        target_answer=answers[f"{item['id']}:target"],
                    )
                    continue
                ops[i] = self._build_op(
                    item=item,
                    answers=answers,
                    targets=targets_by_pid[item["id"]],
                    uncertain=None,
                    proposal_ids=proposal_ids,
                    by_id=by_id,
                    scope_epoch=scope_epoch,
                )

        chunks = [
            undecided[i : i + self.batch_size]
            for i in range(0, len(undecided), self.batch_size)
        ]
        if chunks:
            await asyncio.gather(*(decide(c) for c in chunks))
        result = [op for op in ops if op is not None]
        for op in result:
            if op.op == "ADD":
                op.embedding = embeddings.get(op.proposal_ids[0])
        return result

    def _routed_op(
        self,
        item: dict,
        proposal_ids: set[str],
        by_id: dict[str, MemoryNode],
        scope_epoch: int,
        target_answer: JevAnswer | None = None,
    ) -> ReconcileOp:
        """Independent append for a proposal with no relation target."""
        limited = bool(item.get("valid_until") or item.get("temporal_scope"))
        action = "ADD_TEMPORAL" if limited else "ADD_SUPPLEMENT"
        answers = {
            f"{item['id']}:action": JevAnswer(
                type="choice",
                choice=action,
                confidence=1.0,
                probabilities={action: 1.0},
            )
        }
        if target_answer is not None:
            answers[f"{item['id']}:target"] = target_answer
        op = self._build_op(
            item=item,
            answers=answers,
            targets=[],
            uncertain=None,
            proposal_ids=proposal_ids,
            by_id=by_id,
            scope_epoch=scope_epoch,
        )
        op.decision["routed"] = (
            "jev_target_none" if target_answer is not None else "no_similar_target"
        )
        return op

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
        # ADD_SUPPLEMENT is offered with a selected target (complementary fragment):
        # the target is then context only and no link is written.
        if (
            action not in ("NEED_CONTEXT", "INVALID")
            and not (action == "ADD_SUPPLEMENT" and targets)
            and relation != bool(targets)
        ):
            raise DecisionContextError(f"{action} does not match the selected target")
        if action == "DELETE" and any(t in proposal_ids for t in targets):
            raise DecisionContextError("DELETE cannot target an uncommitted proposal")
        if action == "SKIP" and len(targets) != 1:
            raise DecisionContextError("SKIP requires exactly one covering target")

        # An undecided relation (NEED_CONTEXT, ambiguous MULTIPLE, low confidence) is
        # written as an independent SUPPLEMENT immediately. Nothing is queued for a
        # later review pass: a deferred relation that is never revisited costs more
        # than the occasional extra memory.
        effective = action
        if action == "NEED_CONTEXT" or (
            uncertain is not None and action not in ("RAW_ONLY", "INVALID")
        ) or (relation and (answer.confidence or 0.0) < self.review_confidence):
            effective = "ADD_SUPPLEMENT"
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
        # An unusable validity bound is dropped, never fatal: the candidate text is
        # still a real memory and refusing the whole write loses it.
        try:
            valid_from = _timestamp(item.get("valid_from"))
            valid_until = _timestamp(item.get("valid_until"))
        except DecisionContextError:
            logger.warning("candidate %s: dropping malformed validity bound", pid)
            valid_from = valid_until = None
        if valid_from is not None and valid_until is not None and valid_from > valid_until:
            logger.warning("candidate %s: dropping inverted validity range", pid)
            valid_from = valid_until = None
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
            owner=item.get("owner", ""),
            supersedes=(linked if update_type in ("OVERRIDE", "NEGATE") else []),
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
        )

__all__ = [
    "CoverageError",
    "DecisionContextError",
    "DecisionPlanError",
    "ReconcileOp",
    "Reconciler",
    "build_apply_custom",
    "reflect_custom",
]
