# -*- coding: utf-8 -*-
"""Coding memory data structures — separate schema from chat memories."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

BOUNDARY_SCOPES = ("strict", "project", "user", "global")
BoundaryScope = Literal["strict", "project", "user", "global"]

ReconcileAction = Literal["ADD", "UPDATE", "DELETE", "SKIP"]


@dataclass
class CodingMemoryDraft:
    """LLM extractor output — not yet persisted."""
    task: str
    search_keys: List[str] = field(default_factory=list)
    solution: str = ""
    boundary_envs: str = ""
    boundary_scope: BoundaryScope = "project"
    confidence: float = 0.7
    user_id: str = ""
    agent_id: str = "default_agent"
    workspace_id: Optional[str] = None
    branch: Optional[str] = None
    session_id: Optional[str] = None
    files: List[str] = field(default_factory=list)
    source: str = "auto_extract"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "search_keys": list(self.search_keys),
            "solution": self.solution,
            "boundary_envs": self.boundary_envs,
            "boundary_scope": self.boundary_scope,
            "confidence": self.confidence,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
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
    agent_id: str = "default_agent"
    task: str = ""
    search_keys: List[str] = field(default_factory=list)
    solution: str = ""
    boundary_envs: str = ""
    boundary_scope: BoundaryScope = "project"
    workspace_id: Optional[str] = None
    branch: Optional[str] = None
    session_id: Optional[str] = None
    files: List[str] = field(default_factory=list)
    confidence: float = 0.7
    source: str = "auto_extract"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
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
    draft_idx: Optional[int] = None
    target_memory_id: Optional[str] = None
    reason: Optional[str] = None
