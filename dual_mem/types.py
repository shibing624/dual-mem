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
    """八层记忆分层模型（L0-L7），语义完全对标 hy_memory。

    两条主线：
    - 事实/情境线（episodic，读路径归入 ``NORMAL_LAYERS``）：
      L1 原始 → L2 事实 → L3 摘要 → L5 知识
    - 画像/意图线（profile，读路径归入 ``PROFILE_LAYERS`` / ``PROACTIVE_LAYERS``）：
      L0 基础信息 → L4 身份 → L6 心智模型 → L7 意图

    各层语义（参考 hy_memory）：
    - L0_BASIC_INFO — 基础信息层：姓名/年龄/所在地等结构化基础画像。
    - L1_RAW        — 原始对话层：写入即落、Append-Only 的原始文本。
    - L2_FACT       — 原子事实层：从对话抽取的离散、版本化事实记录。
    - L3_SUMMARY    — 会话摘要层：长文本（≥500 字）压缩出的摘要。
    - L4_IDENTITY   — 身份画像层：用户身份与长期偏好的核心画像。
    - L5_KNOWLEDGE  — 知识图谱层：实体/关系/主题类知识（Graph 层）。
                      注意：dual-mem 当前**未实现**该层 producer，
                      仅在读路径 ``NORMAL_LAYERS`` 中保留，不会有节点被创建。
    - L6_SCHEMA     — 心智模型层：跨证据归纳的抽象行为模式/叙事模板（Graph 层）。
    - L7_INTENTION  — 前瞻意图层：用户未来待触发的具象意图（Graph 层）。

    存储分界（概念上对标 hy_memory 的 VDB/Graph 分界）：L0-L4 属事实/画像主线，
    L5-L7 属高层知识/图主线。dual-mem 当前统一落库于 Chroma（VDB）+ SQLite，
    尚未拆分独立 Graph 存储；L6/L7 以普通节点形式存在。

    每个枚举值经由 ``LAYER_TO_CATEGORY`` 映射到一个 ``Category``。
    """

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
    score: float = field(default=0.0, compare=False)

    @property
    def category(self) -> Category:
        """Routing category derived from the node's layer."""
        return LAYER_TO_CATEGORY[self.layer]

    def to_metadata(self) -> dict:
        """Flatten this node into a storage-friendly metadata dict (lists joined, None encoded)."""
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
            owner=meta.get("owner", ""),
            memory_at=memory_at if memory_at != -1 else None,
            gmt_created=meta["gmt_created"],
            gmt_modified=gmt_modified if gmt_modified != -1 else None,
            embedding=embedding,
            s2_evidence_count=meta["s2_evidence_count"],
            custom=json.loads(meta["custom"]) or None,
        )
