# -*- coding: utf-8 -*-
"""Coding memory data structures — separate schema from chat memories."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

BOUNDARY_SCOPES = ("strict", "project", "user", "global")
BoundaryScope = Literal["strict", "project", "user", "global"]

ReconcileAction = Literal["ADD", "SKIP", "DELETE", "RAW_ONLY"]
UpdateType = Literal[
    "",
    "SUPPLEMENT",
    "OVERRIDE",
    "MERGE",
    "TEMPORAL",
    "NEGATE",
    "CONFLICT",
]
CodingScope = tuple[str, str, str]


class CodingDecisionError(RuntimeError):
    """Incomplete evidence or an invalid executable JEV decision."""


class CodingConcurrentWriteError(RuntimeError):
    """The scope changed after the decision snapshot; reconcile again."""


@dataclass
class CodingMemoryDraft:
    """LLM extractor output — not yet persisted."""

    task: str
    search_keys: list[str] = field(default_factory=list)
    solution: str = ""
    boundary_envs: str = ""
    boundary_scope: BoundaryScope = "project"
    confidence: float = 0.7
    user_id: str = ""
    agent_id: str = ""
    app_id: str = "default"
    workspace_id: str | None = None
    branch: str | None = None
    session_id: str | None = None
    files: list[str] = field(default_factory=list)
    source: str = "auto_extract"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "search_keys": list(self.search_keys),
            "solution": self.solution,
            "boundary_envs": self.boundary_envs,
            "boundary_scope": self.boundary_scope,
            "confidence": self.confidence,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "app_id": self.app_id,
            "workspace_id": self.workspace_id,
            "branch": self.branch,
            "session_id": self.session_id,
            "files": list(self.files),
            "source": self.source,
        }


@dataclass
class CodingMemory:
    """Persisted coding memory (SQLite + VDB dual-layer)."""

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
    session_id: str | None = None
    files: list[str] = field(default_factory=list)
    confidence: float = 0.7
    source: str = "auto_extract"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    is_latest: bool = True
    supersedes: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_latest": self.is_latest,
            "supersedes": list(self.supersedes),
            "superseded_by": list(self.superseded_by),
            "source_ids": list(self.source_ids),
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "app_id": self.app_id,
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
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class CodingReconcileOp:
    """Reconciler decision for one draft."""

    action: ReconcileAction
    draft_idx: int | None = None
    target_memory_id: str | None = None
    update_type: UpdateType = ""
    reason: str | None = None


def coding_scope(memory: CodingMemory | CodingMemoryDraft) -> CodingScope:
    return memory.app_id, memory.user_id, memory.agent_id


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
