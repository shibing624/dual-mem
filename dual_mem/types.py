import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

_LIST_SEP = "\x1f"


class Layer(str, Enum):
    L0_BASIC_INFO = "L0_BASIC_INFO"
    L1_RAW = "L1_RAW"
    L2_FACT = "L2_FACT"
    L3_SUMMARY = "L3_SUMMARY"
    L4_IDENTITY = "L4_IDENTITY"
    L5_KNOWLEDGE = "L5_KNOWLEDGE"
    L6_SCHEMA = "L6_SCHEMA"
    L7_INTENTION = "L7_INTENTION"


class Category(str, Enum):
    raw = "raw"
    fact = "fact"
    summary = "summary"
    profile = "profile"
    knowledge = "knowledge"
    schema = "schema"
    intention = "intention"


class MemoryStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SHADOW = "SHADOW"
    SUPERSEDED = "SUPERSEDED"
    DELETED = "DELETED"


class ReconcileOp(str, Enum):
    ADD = "ADD"
    SUPERSEDE = "SUPERSEDE"
    DELETE = "DELETE"


LAYER_TO_CATEGORY: dict[Layer, Category] = {
    Layer.L0_BASIC_INFO: Category.profile,
    Layer.L1_RAW: Category.raw,
    Layer.L2_FACT: Category.fact,
    Layer.L3_SUMMARY: Category.summary,
    Layer.L4_IDENTITY: Category.profile,
    Layer.L5_KNOWLEDGE: Category.knowledge,
    Layer.L6_SCHEMA: Category.schema,
    Layer.L7_INTENTION: Category.intention,
}


def _now() -> int:
    return int(time.time())


def _new_id() -> str:
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
    memory_at: int | None = None
    gmt_created: int = field(default_factory=_now)
    gmt_modified: int | None = None
    embedding: list[float] | None = None
    s2_evidence_count: int = 0
    score: float = field(default=0.0, compare=False)

    @property
    def category(self) -> Category:
        return LAYER_TO_CATEGORY[self.layer]

    def to_metadata(self) -> dict:
        return {
            "layer": self.layer.value,
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
            "memory_at": self.memory_at if self.memory_at is not None else -1,
            "gmt_created": self.gmt_created,
            "gmt_modified": self.gmt_modified if self.gmt_modified is not None else -1,
            "s2_evidence_count": self.s2_evidence_count,
        }

    @classmethod
    def from_storage(
        cls, content: str, meta: dict, embedding: list[float] | None = None
    ) -> "MemoryNode":
        def _split(value: str) -> list[str]:
            return value.split(_LIST_SEP) if value else []

        speculate = meta["speculate"]
        memory_at = meta["memory_at"]
        gmt_modified = meta["gmt_modified"]
        return cls(
            content=content,
            layer=Layer(meta["layer"]),
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
            memory_at=memory_at if memory_at != -1 else None,
            gmt_created=meta["gmt_created"],
            gmt_modified=gmt_modified if gmt_modified != -1 else None,
            embedding=embedding,
            s2_evidence_count=meta["s2_evidence_count"],
        )
