# -*- coding: utf-8 -*-
"""Procedure memory: reusable engineering experience stored as L1 ``procedure`` nodes.

A procedure lives in the same vector store, scope and version chain as chat facts.
Its structured fields stay in ``custom["procedure"]``; ``content`` is the readable
task + solution text, and the embedding is computed from the task and its
alternative phrasings (the question side a later search matches).
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from dual_mem.sdk_models import ProcedureDetail
from dual_mem.types import Layer, MemoryNode

PROCEDURE_KIND = "procedure"

BOUNDARY_SCOPES = ("strict", "project", "user", "global")
BoundaryScope = Literal["strict", "project", "user", "global"]

ReconcileAction = Literal["ADD", "SKIP", "DELETE", "RAW_ONLY", "INVALID"]
UpdateType = Literal[
    "",
    "SUPPLEMENT",
    "OVERRIDE",
    "MERGE",
    "TEMPORAL",
    "NEGATE",
    "CONFLICT",
]
# A new version of one of these replaces (retires) its target head.
RETIRING_UPDATES = frozenset({"OVERRIDE", "MERGE", "NEGATE"})


class CodingDecisionError(RuntimeError):
    """Incomplete evidence or an invalid executable JEV decision."""


@dataclass
class CodingMemoryDraft:
    """Extractor output — a candidate procedure, not yet decided or persisted."""

    task: str
    search_keys: list[str] = field(default_factory=list)
    solution: str = ""
    boundary_envs: str = ""
    boundary_scope: BoundaryScope = "project"
    confidence: float = 0.7
    workspace_id: str | None = None
    branch: str | None = None
    session_id: str = ""
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "search_keys": list(self.search_keys),
            "solution": self.solution,
            "boundary_envs": self.boundary_envs,
            "boundary_scope": self.boundary_scope,
            "confidence": self.confidence,
            "workspace_id": self.workspace_id,
            "branch": self.branch,
            "session_id": self.session_id,
            "files": list(self.files),
        }

    def detail(self) -> ProcedureDetail:
        return ProcedureDetail(
            task=self.task,
            solution=self.solution,
            search_keys=list(self.search_keys),
            boundary_envs=self.boundary_envs,
            boundary_scope=self.boundary_scope,
            files=list(self.files),
            workspace_id=self.workspace_id,
            branch=self.branch,
            confidence=self.confidence,
        )

    def content(self) -> str:
        """Readable stored text: what the agent sees when the procedure is recalled."""
        parts = [self.task.strip(), self.solution.strip()]
        if self.boundary_envs.strip():
            parts.append(self.boundary_envs.strip())
        return "\n".join(parts)

    def index_text(self) -> str:
        """Embedding input: the task and its alternative phrasings."""
        return "\n".join([self.task, *self.search_keys])


@dataclass
class CodingMemory:
    """Read view of one stored procedure node (current head or history version)."""

    memory_id: str
    user_id: str
    agent_id: str = ""
    app_id: str = "default"
    task: str = ""
    search_keys: list[str] = field(default_factory=list)
    solution: str = ""
    boundary_envs: str = ""
    boundary_scope: BoundaryScope = "project"
    workspace_id: str | None = None
    branch: str | None = None
    session_id: str = ""
    files: list[str] = field(default_factory=list)
    confidence: float = 0.7
    created_at: datetime | None = None
    revision: int = 0
    is_latest: bool = True
    supersedes: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_node(cls, node: MemoryNode) -> "CodingMemory":
        if node.kind != PROCEDURE_KIND or node.layer != Layer.L1_FACT:
            raise CodingDecisionError(f"node {node.node_id} is not a procedure")
        custom = node.custom or {}
        detail = custom["procedure"]
        return cls(
            memory_id=node.node_id,
            user_id=node.user_id,
            agent_id=node.agent_id,
            app_id=node.app_id,
            task=detail["task"],
            search_keys=list(detail["search_keys"]),
            solution=detail["solution"],
            boundary_envs=detail["boundary_envs"],
            boundary_scope=detail["boundary_scope"],
            workspace_id=detail["workspace_id"],
            branch=detail["branch"],
            session_id=node.session_id,
            files=list(detail["files"]),
            confidence=float(detail["confidence"]),
            created_at=datetime.fromtimestamp(node.gmt_created, tz=timezone.utc),
            revision=node.revision,
            is_latest=node.is_latest,
            supersedes=list(node.supersedes),
            superseded_by=list(node.superseded_by),
            source_ids=list(custom.get("source_ids", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "task": self.task,
            "search_keys": list(self.search_keys),
            "solution": self.solution,
            "boundary_envs": self.boundary_envs,
            "boundary_scope": self.boundary_scope,
            "workspace_id": self.workspace_id,
            "branch": self.branch,
            "session_id": self.session_id,
            "files": list(self.files),
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_latest": self.is_latest,
            "supersedes": list(self.supersedes),
            "superseded_by": list(self.superseded_by),
            "source_ids": list(self.source_ids),
        }


@dataclass
class CodingReconcileOp:
    """Reconciler decision for one draft."""

    action: ReconcileAction
    draft_idx: int
    target_memory_id: str | None = None
    update_type: UpdateType = ""
    reason: str = ""
    confidence: float = 0.0


def version_boundary(
    memory: CodingMemory | CodingMemoryDraft,
) -> tuple[str, str | None, str | None, str | None]:
    """Structural retirement boundary; JEV compares semantic environment details."""
    return (
        memory.boundary_scope,
        memory.workspace_id,
        memory.branch,
        memory.session_id if memory.boundary_scope == "strict" else None,
    )


def procedure_detail(node: MemoryNode) -> ProcedureDetail | None:
    """Structured fields of a procedure node; None for every other memory kind."""
    if node.kind != PROCEDURE_KIND:
        return None
    detail = (node.custom or {})["procedure"]
    return ProcedureDetail(
        task=detail["task"],
        solution=detail["solution"],
        search_keys=list(detail["search_keys"]),
        boundary_envs=detail["boundary_envs"],
        boundary_scope=detail["boundary_scope"],
        files=list(detail["files"]),
        workspace_id=detail["workspace_id"],
        branch=detail["branch"],
        confidence=float(detail["confidence"]),
    )


def revised_procedure(node: MemoryNode, solution: str) -> CodingMemoryDraft:
    """The procedure of ``node`` with an explicitly corrected solution.

    ``MemoryClient.update`` replaces the solution of a procedure; its task, search
    keys and boundary stay, so the new version keeps its question-side index.
    """
    detail = (node.custom or {})["procedure"]
    return CodingMemoryDraft(
        task=detail["task"],
        search_keys=list(detail["search_keys"]),
        solution=solution,
        boundary_envs=detail["boundary_envs"],
        boundary_scope=detail["boundary_scope"],
        confidence=float(detail["confidence"]),
        workspace_id=detail["workspace_id"],
        branch=detail["branch"],
        session_id=node.session_id,
        files=list(detail["files"]),
    )
