"""JEV owns source support, value, action and current target."""

from typing import Any

from dual_mem.coding.judge import evaluate_decisions
from dual_mem.coding.types import (
    CodingDecisionError,
    CodingMemory,
    CodingMemoryDraft,
    CodingReconcileOp,
    coding_scope,
    version_boundary,
)
from dual_mem.providers.jev import DecisionProvider

_ACTIONS = {
    "ADD_SUPPLEMENT": "Independent learning or a distinct occurrence; append. Initial solution without a baseline belongs here.",
    "ADD_OVERRIDE": "Confirmed replacement of the same task and boundary. Append a complete version. Returning to an ancestor solution overrides the current head, never SKIP.",
    "ADD_MERGE": "Complementary fragments of ONE solution; the new version must preserve every unaffected fact, number and constraint.",
    "ADD_TEMPORAL": "Explicit limited exception that coexists with the long-term solution. Do not retire the baseline.",
    "ADD_NEGATE": "Explicit termination of an existing solution.",
    "ADD_CONFLICT": "Competing simultaneous solutions with no explicit change or authority. Keep both; do not retire either.",
    "SKIP": "A CURRENT head already fully covers this claim. The incoming source, the draft itself and history NEVER mean known. NONE is not coverage.",
    "RAW_ONLY": "Supported candidate has no durable persistence value; keep source.",
    "DELETE": "Explicit retraction; archive the current head, do not erase it.",
    "INVALID": "Candidate invents facts or incorrectly attributes source evidence.",
    "NEED_CONTEXT": "Insufficient evidence or a lossless version is unavailable.",
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
_WITHOUT_TARGET = (
    "ADD_SUPPLEMENT",
    "ADD_TEMPORAL",
    "RAW_ONLY",
    "INVALID",
    "NEED_CONTEXT",
)
_NEEDS_TARGET = set(_WITH_TARGET) - {"INVALID", "NEED_CONTEXT"}


def _action_question(draft_idx: int, *, has_target: bool) -> dict:
    if has_target:
        instructions = (
            f"Draft {draft_idx} already has one selected current head in the same version boundary. "
            "Its environment constraints must describe the same solution context. "
            "Choose only a relation to that head: SKIP if it fully covers the draft, "
            "ADD_OVERRIDE or ADD_MERGE or ADD_NEGATE to replace or end it, "
            "ADD_CONFLICT to keep both, DELETE to archive it. "
            "INVALID if the draft invents facts. NEED_CONTEXT if a lossless version is unavailable. "
            "Do not append an independent solution."
        )
        keys = _WITH_TARGET
    else:
        instructions = (
            f"Draft {draft_idx} has no selected current head. "
            "There is nothing to update, cover, negate, conflict with, or delete. "
            "ADD_SUPPLEMENT appends an independent solution. "
            "ADD_TEMPORAL is a limited exception and does not link a baseline. "
            "RAW_ONLY keeps only the source. INVALID if the draft invents facts. "
            "NEED_CONTEXT if the evidence is insufficient."
        )
        keys = _WITHOUT_TARGET
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {key: _ACTIONS[key] for key in keys},
    }


class CodingMemoryReconciler:
    def __init__(self, *, jev: DecisionProvider, candidate_limit: int = 64) -> None:
        if not 1 <= candidate_limit <= 253:
            raise ValueError("coding candidate_limit must be in [1,253]")
        self.jev = jev
        self.candidate_limit = candidate_limit

    async def reconcile(
        self,
        drafts: list[CodingMemoryDraft],
        existing: list[CodingMemory],
        *,
        source_messages: list[dict],
        history_versions: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[CodingReconcileOp]:
        if not source_messages:
            raise CodingDecisionError("coding reconciliation requires source evidence")
        histories = history_versions or {}
        if drafts and len({coding_scope(d) for d in drafts}) != 1:
            raise CodingDecisionError("coding candidates cross tenant scope")
        scope = coding_scope(drafts[0]) if drafts else None
        current = []
        by_id = {}
        for mem in existing:
            if not mem.is_latest or (scope is not None and coding_scope(mem) != scope):
                raise CodingDecisionError(
                    "coding current target crosses scope or is retired"
                )
            if mem.memory_id in by_id or mem.memory_id in ("NONE", "MULTIPLE"):
                raise CodingDecisionError("invalid coding current target ID")
            if mem.supersedes and mem.memory_id not in histories:
                raise CodingDecisionError("missing coding history_versions")
            history = histories.get(mem.memory_id, [])
            ancestors = {m["memory_id"]: m for m in history}
            if len(ancestors) != len(history):
                raise CodingDecisionError("duplicate coding history_versions")
            pending = [(old_id, mem.memory_id) for old_id in mem.supersedes]
            visited = set()
            while pending:
                old_id, child_id = pending.pop()
                old = ancestors.get(old_id)
                if old is None:
                    raise CodingDecisionError(
                        "missing coding history_versions ancestor"
                    )
                if (
                    old.get("is_latest", True)
                    or (old.get("app_id"), old.get("user_id"), old.get("agent_id"))
                    != coding_scope(mem)
                    or child_id not in old.get("superseded_by", [])
                ):
                    raise CodingDecisionError(
                        "invalid coding history_versions ancestor"
                    )
                if old_id not in visited:
                    visited.add(old_id)
                    pending.extend((oid, old_id) for oid in old.get("supersedes", []))
            if visited != set(ancestors):
                raise CodingDecisionError("unlinked coding history_versions")
            current.append(
                dict(mem.to_dict(), history_versions=histories.get(mem.memory_id, []))
            )
            by_id[mem.memory_id] = mem

        if not drafts:
            # Same as the chat path: no candidates keeps the source RAW_ONLY without
            # spending a model call on whether the extractor missed something.
            return []
        proposals = [dict(d.to_dict(), id=i) for i, d in enumerate(drafts)]
        if len(current) > self.candidate_limit:
            selected = []
            for start in range(0, len(current), self.candidate_limit):
                page = current[start : start + self.candidate_limit]
                answers = await evaluate_decisions(
                    self.jev,
                    state={
                        "source": source_messages,
                        "drafts": proposals,
                        "current": page,
                    },
                    questions={
                        m["memory_id"]: {
                            "type": "choice",
                            "instructions": (
                                "Is this current head or its history potentially relevant "
                                "to ANY draft as coverage, update or boundary baseline? "
                                "Use INCLUDE when unsure; compare complete solutions."
                            ),
                            "criteria": {
                                "INCLUDE": "Relevant",
                                "EXCLUDE": "Clearly unrelated",
                            },
                        }
                        for m in page
                    },
                )
                selected.extend(
                    m for m in page if answers[m["memory_id"]].choice == "INCLUDE"
                )
            if len(selected) > self.candidate_limit:
                raise CodingDecisionError(
                    "coding selected material exceeds candidate budget"
                )
            current = selected

        # No coverage gate (as in the chat path): a rejected extraction would discard
        # every candidate of the write; extraction runs once and the source is kept.
        questions: dict[str, dict[str, Any]] = {}
        # A relation can only link heads in the draft's own version boundary
        # (structured metadata), so other boundaries are never offered as targets.
        heads = {
            i: {
                m["memory_id"]: m["task"]
                for m in current
                if version_boundary(by_id[m["memory_id"]]) == version_boundary(draft)
            }
            for i, draft in enumerate(drafts)
        }
        for i in range(len(drafts)):
            if heads[i]:
                questions[f"{i}:target"] = {
                    "type": "choice",
                    "instructions": (
                        f"Select the one CURRENT head for a relation to draft {i}, "
                        "or NONE when the draft is an independent solution, a limited "
                        "exception, has no durable value, or has independent "
                        "environment constraints. Never select an ancestor or another "
                        "incoming draft. "
                        "Choose MULTIPLE only when the candidates need per-head "
                        "adjudication; expansion must resolve to exactly one head."
                    ),
                    "criteria": {
                        **heads[i],
                        "NONE": "No current head",
                        "MULTIPLE": "Needs per-head adjudication",
                    },
                }
            else:
                questions[f"{i}:action"] = _action_question(i, has_target=False)
        state = {
            "source": source_messages,
            "drafts": proposals,
            "current": current,
        }
        answers = await evaluate_decisions(self.jev, state=state, questions=questions)
        chosen: dict[int, str | None] = {}
        for i in range(len(drafts)):
            if not heads[i]:
                chosen[i] = None
                continue
            target = answers[f"{i}:target"].choice
            if target == "MULTIPLE":
                include = {
                    f"{i}:{nid}": {
                        "type": "choice",
                        "instructions": (
                            f"Is current head {nid} required for draft {i}?"
                        ),
                        "criteria": {
                            "INCLUDE": "Required",
                            "EXCLUDE": "Unrelated",
                            "UNSURE": "Insufficient evidence",
                        },
                    }
                    for nid in heads[i]
                }
                picked = await evaluate_decisions(
                    self.jev, state=state, questions=include
                )
                if any(item.choice == "UNSURE" for item in picked.values()):
                    raise CodingDecisionError("uncertain coding targets")
                matched = [
                    nid
                    for nid in heads[i]
                    if picked[f"{i}:{nid}"].choice == "INCLUDE"
                ]
                if len(matched) != 1:
                    raise CodingDecisionError(
                        "coding relation requires exactly one current head"
                    )
                target = matched[0]
            chosen[i] = None if target == "NONE" else target
        action_questions = {
            f"{i}:action": _action_question(i, has_target=chosen[i] is not None)
            for i in range(len(drafts))
            if f"{i}:action" not in answers
        }
        if action_questions:
            answers.update(
                await evaluate_decisions(
                    self.jev,
                    state={**state, "selected_targets": chosen},
                    questions=action_questions,
                )
            )
        ops = []
        for i in range(len(drafts)):
            # evaluate_decisions already restricts the choice to the offered criteria.
            action = answers[f"{i}:action"].choice
            target = chosen[i]
            if action in ("INVALID", "NEED_CONTEXT"):
                raise CodingDecisionError(
                    f"coding JEV decision {action} for draft {i}"
                )
            ops.append(
                CodingReconcileOp(
                    action="ADD" if action.startswith("ADD_") else action,
                    update_type=(
                        action.removeprefix("ADD_") if action.startswith("ADD_") else ""
                    ),
                    draft_idx=i,
                    target_memory_id=target if action in _NEEDS_TARGET else None,
                    reason=action,
                )
            )
        return ops
