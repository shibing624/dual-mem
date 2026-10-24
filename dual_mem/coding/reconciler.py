"""JEV owns source support, value, action and current target."""

from typing import Any

from dual_mem.coding.judge import evaluate_decisions
from dual_mem.coding.types import (
    CodingDecisionError,
    CodingMemory,
    CodingMemoryDraft,
    CodingReconcileOp,
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
        current = []
        by_id: dict[str, CodingMemory] = {}
        for mem in existing:
            if not mem.is_latest:
                raise CodingDecisionError("coding current target is retired")
            if mem.memory_id in by_id or mem.memory_id in ("NONE", "MULTIPLE"):
                raise CodingDecisionError("invalid coding current target ID")
            # The version store owns chain integrity; the head shows its ancestors so
            # a return to an earlier solution is recognised as an override.
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
            reason = action
            if action == "NEED_CONTEXT":
                # As in the chat path: an undecided relation is written now as an
                # independent version instead of failing the whole write.
                action, target = "ADD_SUPPLEMENT", None
                reason = "NEED_CONTEXT: kept as an independent version"
            if action == "INVALID":
                # Rejects only this draft; the other drafts and the source stay.
                ops.append(
                    CodingReconcileOp(
                        action="INVALID",
                        draft_idx=i,
                        reason=reason,
                        confidence=float(answers[f"{i}:action"].confidence or 0.0),
                    )
                )
                continue
            ops.append(
                CodingReconcileOp(
                    action="ADD" if action.startswith("ADD_") else action,
                    update_type=(
                        action.removeprefix("ADD_") if action.startswith("ADD_") else ""
                    ),
                    draft_idx=i,
                    target_memory_id=target if action in _NEEDS_TARGET else None,
                    reason=reason,
                    confidence=float(answers[f"{i}:action"].confidence or 0.0),
                )
            )
        return ops
