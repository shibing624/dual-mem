# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Core domain types: Layer/Category/MemoryStatus/ReconcileOp enums and the
MemoryNode dataclass with its storage (de)serialization helpers.
"""
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

_LIST_SEP = "\x1f"


class Layer(str, Enum):
    """Four processing layers; content kinds do not create additional layers.

    Stored values use only the four canonical layer names.
    """

    L0_RAW = "L0_RAW"
    L1_FACT = "L1_FACT"
    L2_PROFILE = "L2_PROFILE"
    L3_DERIVED = "L3_DERIVED"


class Category(str, Enum):
    raw = "raw"
    fact = "fact"
    summary = "summary"
    profile = "profile"
    schema = "schema"
    intention = "intention"
    preference = "preference"


class MemoryStatus(str, Enum):
    ACTIVE = "ACTIVE"


class ReconcileOp(str, Enum):
    ADD = "ADD"
    SKIP = "SKIP"
    DELETE = "DELETE"
    RAW_ONLY = "RAW_ONLY"
    INVALID = "INVALID"


LAYER_TO_CATEGORY: dict[Layer, Category] = {
    Layer.L0_RAW: Category.raw,
    Layer.L1_FACT: Category.fact,
    Layer.L2_PROFILE: Category.profile,
    Layer.L3_DERIVED: Category.schema,
}


def _now() -> int:
    """Return the current Unix timestamp in seconds."""
    return int(time.time())


def _new_id() -> str:
    """Generate a fresh random UUID4 string."""
    return str(uuid.uuid4())


@dataclass
class MemoryNode:
    content: str
    layer: Layer
    app_id: str
    user_id: str
    agent_id: str = ""
    session_id: str = ""
    tags: list[str] = field(default_factory=list)
    node_id: str = field(default_factory=_new_id)
    status: MemoryStatus = MemoryStatus.ACTIVE
    supersedes: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)
    is_latest: bool = True
    speculate: str | None = None
    owner: str = ""  # "user" | "agent" — who said it; empty = unknown
    memory_at: int | None = None
    gmt_created: int = field(default_factory=_now)
    gmt_modified: int | None = None
    embedding: list[float] | None = None
    s2_evidence_count: int = 0
    custom: dict | None = None
    kind: str = ""
    revision: int = 0
    valid_from: int | None = None
    valid_until: int | None = None
    score: float = field(default=0.0, compare=False)

    @property
    def category(self) -> Category:
        """Routing category derived from the node's layer."""
        kind = self.kind or (self.custom or {}).get("kind")
        if kind in Category._value2member_map_:
            return Category(kind)
        return LAYER_TO_CATEGORY[self.layer]

    def to_metadata(self) -> dict:
        """Flatten this node into a storage-friendly metadata dict (lists joined, None encoded)."""
        return {
            "layer": self.layer.value,
            "layer_schema_version": 4,
            "kind": self.kind,
            "revision": self.revision,
            "valid_from": self.valid_from if self.valid_from is not None else -1,
            "valid_until": self.valid_until if self.valid_until is not None else -1,
            "app_id": self.app_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "tags": _LIST_SEP.join(self.tags),
            "node_id": self.node_id,
            "status": self.status.value,
            "supersedes": _LIST_SEP.join(self.supersedes),
            "superseded_by": _LIST_SEP.join(self.superseded_by),
            "is_latest": self.is_latest,
            "speculate": self.speculate if self.speculate is not None else "",
            "owner": self.owner,
            "memory_at": self.memory_at if self.memory_at is not None else -1,
            "gmt_created": self.gmt_created,
            "gmt_modified": self.gmt_modified if self.gmt_modified is not None else -1,
            "s2_evidence_count": self.s2_evidence_count,
            "custom": json.dumps(self.custom or {}, ensure_ascii=False),
        }

    @classmethod
    def from_storage(
        cls, content: str, meta: dict, embedding: list[float] | None = None
    ) -> "MemoryNode":
        """Reconstruct a MemoryNode from stored content/metadata (inverse of to_metadata)."""

        if meta.get("layer_schema_version") != 4:
            raise ValueError("unsupported memory schema; use a new storage directory")

        def _split(value: str) -> list[str]:
            return value.split(_LIST_SEP) if value else []

        speculate = meta["speculate"]
        memory_at = meta["memory_at"]
        gmt_modified = meta["gmt_modified"]
        return cls(
            content=content,
            layer=Layer(meta["layer"]),
            kind=meta["kind"],
            revision=int(meta["revision"]),
            valid_from=None if meta["valid_from"] == -1 else meta["valid_from"],
            valid_until=(None if meta["valid_until"] == -1 else meta["valid_until"]),
            app_id=meta["app_id"],
            user_id=meta["user_id"],
            agent_id=meta["agent_id"],
            session_id=meta["session_id"],
            tags=_split(meta["tags"]),
            node_id=meta["node_id"],
            status=MemoryStatus(meta["status"]),
            supersedes=_split(meta["supersedes"]),
            superseded_by=_split(meta["superseded_by"]),
            is_latest=meta["is_latest"],
            speculate=speculate if speculate != "" else None,
            owner=meta["owner"],
            memory_at=memory_at if memory_at != -1 else None,
            gmt_created=meta["gmt_created"],
            gmt_modified=gmt_modified if gmt_modified != -1 else None,
            embedding=embedding,
            s2_evidence_count=meta["s2_evidence_count"],
            custom=json.loads(meta["custom"]) or None,
        )
