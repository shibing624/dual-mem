"""JEV owns coverage, source support, value, action and current target."""

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
    "ADD": "Independent compatible learning or a distinct event; append.",
    "UPDATE": (
        "Confirmed change to the same task and boundary. Append a complete version "
        "preserving every unaffected fact of the current target. If any fact would "
        "be lost, choose NEED_CONTEXT; never retire an independent environment."
    ),
    "SKIP": (
        "A CURRENT head fully covers the supported valuable candidate. Matching "
        "incoming source, the candidate itself or history NEVER means known."
    ),
    "RAW_ONLY": "Supported candidate has no durable persistence value; keep source.",
    "INVALID": "Candidate invents facts or incorrectly attributes source evidence.",
    "NEED_CONTEXT": "Insufficient evidence or a lossless version is unavailable.",
}


class CodingMemoryReconciler:
    def __init__(self, *, jev: DecisionProvider, candidate_limit: int = 64) -> None:
        if not 1 <= candidate_limit <= 254:
            raise ValueError("coding candidate_limit must be in [1,254]")
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

        proposals = [dict(d.to_dict(), id=i) for i, d in enumerate(drafts)]
        if drafts and len(current) > self.candidate_limit:
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

        questions = {
            "coverage": {
                "type": "choice",
                "instructions": (
                    "Check extraction against FULL source including tool arguments "
                    "and results. Every potentially durable fact, constraint, number, "
                    "changed solution and distinct event must be represented without "
                    "losing qualifications. Empty extraction is RAW_ONLY only when "
                    "there is no durable information. Existing knowledge and history "
                    "are not reasons to omit source facts from candidates."
                ),
                "criteria": (
                    {"VALID": "Complete candidates", "MISSED": "Facts omitted"}
                    if drafts
                    else {
                        "RAW_ONLY": "No durable information",
                        "MISSED": "Facts omitted",
                    }
                ),
            }
        }
        targets = {m["memory_id"]: m["task"] for m in current}
        for i in range(len(drafts)):
            questions[f"{i}:action"] = {
                "type": "choice",
                "instructions": (
                    f"Decide draft {i} using NEW source for support only. First "
                    "verify faithfulness, then value, then knownness/change using "
                    "current heads and history_versions. Returning to an ancestor "
                    "solution is UPDATE of the current head, never SKIP. Equal "
                    "task names do not mean equal solutions. Preserve independent "
                    "workspace/branch/session/environment boundaries. Drafts are "
                    "proposals, not stored knowledge."
                ),
                "criteria": _ACTIONS,
            }
            questions[f"{i}:target"] = {
                "type": "choice",
                "instructions": (
                    f"Choose the CURRENT head for UPDATE/SKIP of draft {i}; "
                    "never an ancestor or another incoming draft. Choose NONE "
                    "for ADD/RAW_ONLY and MULTIPLE if one target cannot suffice."
                ),
                "criteria": {
                    **targets,
                    "NONE": "No target",
                    "MULTIPLE": "Needs expanded context",
                },
            }
        answers = await evaluate_decisions(
            self.jev,
            state={"source": source_messages, "drafts": proposals, "current": current},
            questions=questions,
        )
        if answers["coverage"].choice == "MISSED":
            raise CodingDecisionError("coding extraction missed source coverage")
        ops = []
        for i, draft in enumerate(drafts):
            action = answers[f"{i}:action"].choice
            target = answers[f"{i}:target"].choice
            if action in ("INVALID", "NEED_CONTEXT") or target == "MULTIPLE":
                raise CodingDecisionError(
                    f"coding JEV decision {action}/{target} for draft {i}"
                )
            if action in ("UPDATE", "SKIP"):
                if target not in targets:
                    raise CodingDecisionError(
                        "coding decision requires a current target"
                    )
                if action == "UPDATE" and version_boundary(draft) != version_boundary(
                    by_id[target]
                ):
                    raise CodingDecisionError(
                        "coding retirement crosses version boundary"
                    )
            elif target != "NONE":
                raise CodingDecisionError(
                    "coding non-retiring decision requires NONE target"
                )
            ops.append(
                CodingReconcileOp(
                    action=action,
                    draft_idx=i,
                    target_memory_id=None if target == "NONE" else target,
                    reason=action,
                )
            )
        return ops
