# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SDK-level dataclass models returned by MemoryClient. The SDK keeps strongly
typed objects internally (IDE-friendly attribute access, refactor-safe) and only flattens to
plain dicts at REST/MCP/CLI boundaries via .to_dict().

Time fields (``memory_at`` / ``gmt_created`` / ``gmt_modified``) are **Unix timestamps in
seconds**, UTC. Prefer ``datetime.fromtimestamp(ts, tz=timezone.utc)`` when displaying.

Write-side example::

    import time
    from datetime import datetime, timezone

    memory_at = int(time.time())  # now
    # or: int(datetime(2023, 5, 15, tzinfo=timezone.utc).timestamp())
    await client.add(content="...", user_id="u", memory_at=memory_at)

Read-side example::

    ts = item.memory_at or item.gmt_created
    if ts:
        label = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _omit_none(data: dict) -> dict:
    """Drop keys whose value is None, leaving empty lists/dicts untouched."""
    return {k: v for k, v in data.items() if v is not None}


@dataclass
class ChatMessage:
    """One turn of a multi-turn write input."""

    # 角色：user / assistant / system（与 OpenAI messages 一致）
    role: str
    # 该轮文本内容
    content: str

    def to_dict(self) -> dict:
        """Plain dict form so callers can treat ChatMessage and {role,content} dicts identically."""
        return {"role": self.role, "content": self.content}


@dataclass
class MemoryDecision:
    """Executable Jev decision returned to the caller.

    ``action`` is ``ADD`` / ``SKIP`` (already known) / ``RAW_ONLY`` (no durable
    increment) / ``DELETE`` / ``INVALID`` (unsupported candidate, not written).
    An undecided relation is written as an independent SUPPLEMENT immediately; the
    SDK keeps no deferred review queue.
    """

    action: str
    update_type: str | None = None
    targets: list[str] = field(default_factory=list)
    proposal_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict:
        return _omit_none(asdict(self))


WRITE_STATUSES = ("committed", "known", "raw_only")


def write_status(decisions: list[MemoryDecision]) -> str:
    """Summarize decisions: committed > known > raw_only."""
    actions = {d.action for d in decisions}
    if actions & {"ADD", "DELETE"}:
        return "committed"
    if "SKIP" in actions:
        return "known"
    return "raw_only"


@dataclass
class CommitResult:
    """Jev-backed decision on whether a raw write should produce derived memories."""

    # 是否追加结构化版本；False 时保留 L0_RAW 来源
    passed: bool
    # 可读的提交/跳过原因
    reason: str = ""
    decisions: list[MemoryDecision] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Flatten this commit decision into a dict for logging or contract responses."""
        return _omit_none(asdict(self))


@dataclass
class WriteResult:
    """Outcome of a single MemoryClient.add call."""

    # 写入是否成功（仅保留 L0_RAW 时仍为 True，见 is_ephemeral）
    success: bool
    # 本次写入的 L0_RAW 节点 id（主 raw id）
    memory_id: str
    # 请求追踪 id（日志/REST 对齐）
    request_id: str
    # 端到端处理耗时（毫秒）
    processing_time_ms: float
    # Jev 是否提交结构化版本或归档动作
    commit_passed: bool = True
    # 本次写入产生的 L1/L2（digest 为 L3）版本数量
    extracted_count: int = 0
    # 除 L0 raw 外写入的节点 id 列表
    extra_node_ids: list[str] = field(default_factory=list)
    # True 表示 Jev 判定 RAW_ONLY，仅保留来源
    is_ephemeral: bool = False
    decisions: list[MemoryDecision] = field(default_factory=list)
    # committed / known / raw_only（见 ``write_status``）
    status: str = "committed"
    # REST 风格错误码；成功时为 None
    error_code: int | None = None
    # 错误描述；成功时为 None
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Flatten the write result into the contract dict (None fields omitted)."""
        return _omit_none(asdict(self))


@dataclass
class EvolutionItem:
    """One version inside a memory evolution chain (index 0 = current head, newest first)."""

    # 该版本节点 id
    node_id: str
    # 该版本文本
    content: str
    # 存储层代号，如 L1_FACT / L2_PROFILE
    layer: str
    # 语义时间：用户说这话/事件发生的 Unix 秒（UTC）；未知为 None
    memory_at: int | None = None
    # 入库时间：节点写入存储的 Unix 秒（UTC）
    gmt_created: int | None = None
    # 可选推测/标注（reconcile 产物）
    speculate: str | None = None

    def to_dict(self) -> dict:
        """Serialize a single chain entry as a dict, omitting null fields."""
        return _omit_none(asdict(self))


@dataclass
class MemoryItem:
    """One scored memory returned from search / get / list."""

    # 节点 id（与 storage 中 node_id 一致）
    memory_id: str
    # 当前有效内容（演化链上为 is_latest head 的 content）
    content: str
    # 读侧路由类别：profile / fact / identity / schema / intention / raw 等
    category: str
    # 融合排序分，越高越靠前
    score: float
    # 标签列表（可能为空）
    tags: list[str] = field(default_factory=list)
    # 来源会话 id；benchmark 可据此核验是否召回 gold evidence session
    session_id: str = ""
    # 语义时间 Unix 秒（UTC）：对话/session 时间；write 时 memory_at= 传入
    memory_at: int | None = None
    # 入库 Unix 秒（UTC）：首次写入存储的时间
    gmt_created: int | None = None
    # 最后修改 Unix 秒（UTC）；未更新过可为 None
    gmt_modified: int | None = None
    # 同主题演化历史（新→旧）；无链时为 None
    evolution_chain: list[EvolutionItem] | None = None
    # 仅 MERGE 链头：新旧 memory_at 并集；OVERRIDE 为空
    merged_timestamps: list[int] | None = None
    # 写侧关系：OVERRIDE / MERGE / SUPPLEMENT / …；无则 None
    update_type: str | None = None
    # 源 L0_RAW 节点 id（extract / reconcile 写入 custom.source_node_id）
    source_node_id: str | None = None

    layer: str = ""
    kind: str = ""
    is_latest: bool = True
    revision: int = 0
    source_ids: list[str] = field(default_factory=list)
    # Schema only: the retrieved memories (in this result) that this pattern explains.
    supporting_ids: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    valid_from: int | None = None
    valid_until: int | None = None
    # 直接被本版本取代的上一版本（一跳）；search(include_relations=True) 填充
    previous: list[EvolutionItem] | None = None
    # conflicts_with 指向的、仍然有效的对立版本内容
    conflicts: list[EvolutionItem] | None = None
    owner: str = ""

    def to_dict(self) -> dict:
        """Serialize this memory item with its (optional) evolution chain expanded."""
        out: dict[str, Any] = {
            "memory_id": self.memory_id,
            "content": self.content,
            "category": self.category,
            "score": round(self.score, 4),
            "tags": list(self.tags),
            "session_id": self.session_id,
            "layer": self.layer,
            "kind": self.kind,
            "is_latest": self.is_latest,
            "revision": self.revision,
            "source_ids": list(self.source_ids),
            "supporting_ids": list(self.supporting_ids),
            "supersedes": list(self.supersedes),
            "conflicts_with": list(self.conflicts_with),
        }
        if self.valid_from is not None:
            out["valid_from"] = self.valid_from
        if self.owner:
            out["owner"] = self.owner
        if self.valid_until is not None:
            out["valid_until"] = self.valid_until
        if self.memory_at is not None:
            out["memory_at"] = self.memory_at
        if self.gmt_created is not None:
            out["gmt_created"] = self.gmt_created
        if self.gmt_modified is not None:
            out["gmt_modified"] = self.gmt_modified
        if self.evolution_chain:
            out["evolution_chain"] = [item.to_dict() for item in self.evolution_chain]
        if self.merged_timestamps:
            out["merged_timestamps"] = list(self.merged_timestamps)
        if self.update_type:
            out["update_type"] = self.update_type
        if self.source_node_id:
            out["source_node_id"] = self.source_node_id
        if self.previous:
            out["previous"] = [item.to_dict() for item in self.previous]
        if self.conflicts:
            out["conflicts"] = [item.to_dict() for item in self.conflicts]
        return out

    def to_search_result(self) -> dict[str, Any]:
        """Flat search hit for external QA / benchmark pipelines (current fact only).

        ``content`` is always the evolution-chain head (``is_latest``). Superseded
        versions are omitted so downstream LLMs are not confused by stale values.
        """
        ts = self.memory_at or self.gmt_created
        created_at = ""
        if ts:
            created_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        return {
            "memory": self.content,
            "score": self.score,
            "id": self.memory_id,
            "memory_id": self.memory_id,
            "created_at": created_at,
            "category": self.category,
            "session_id": self.session_id,
        }


@dataclass
class SearchMemories:
    """Recall groups from the reader (routing, not storage layers).

    System 2 schemas live in their own ``schema`` group. They are selected after
    the System 1 groups are final, so they never take an evidence slot; each carries
    ``supporting_ids`` — the retrieved memories it explains.
    """

    # 画像路：L2/L3 等身份与 schema
    profile: list[MemoryItem] = field(default_factory=list)
    # 主动路：L1/L3 意图/计划（intention_limit=0 时常为空）
    proactive: list[MemoryItem] = field(default_factory=list)
    # 常规路：L1 facts 等事实、摘要与身份记忆
    normal: list[MemoryItem] = field(default_factory=list)
    # 原文路：L0_RAW 低权重兜底（raw_limit=0 时为空）
    raw: list[MemoryItem] = field(default_factory=list)
    # 模式路：System 2 行为 schema（独立配额，不占证据名额）
    schema: list[MemoryItem] = field(default_factory=list)
    # 画像目录：System 2 按领域选出的当前事实原文（与问题无关）
    catalog: list[MemoryItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Flatten each route's items into plain dicts."""
        return {
            "profile": [item.to_dict() for item in self.profile],
            "proactive": [item.to_dict() for item in self.proactive],
            "normal": [item.to_dict() for item in self.normal],
            "raw": [item.to_dict() for item in self.raw],
            "schema": [item.to_dict() for item in self.schema],
            "catalog": [item.to_dict() for item in self.catalog],
        }

    def flatten(self, *, limit: int | None = None) -> list[MemoryItem]:
        """System 1 evidence ordered by score (``limit`` applies), then schemas, then catalog.

        Schema scores rank schemas among themselves and are never compared with
        evidence scores, so System 2 groups are appended after the bounded evidence.
        """
        items = [*self.profile, *self.proactive, *self.normal, *self.raw]
        items.sort(key=lambda x: x.score, reverse=True)
        unique: list[MemoryItem] = []
        seen: set[str] = set()
        for item in items:
            if item.memory_id in seen:
                continue
            seen.add(item.memory_id)
            unique.append(item)
        if limit is not None and limit >= 0:
            unique = unique[:limit]
        return [*unique, *(s for s in [*self.schema, *self.catalog] if s.memory_id not in seen)]

    def to_search_results(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Current-state search hits for QA pipelines (see ``MemoryItem.to_search_result``)."""
        return [item.to_search_result() for item in self.flatten(limit=limit)]


@dataclass
class SearchResult:
    """Outcome of MemoryClient.search."""

    # 检索是否成功
    success: bool
    # 请求追踪 id
    request_id: str
    # 三路分组结果
    memories: SearchMemories
    # 端到端处理耗时（毫秒）
    processing_time_ms: float
    # debug=True 时的读管线 trace；正常调用为 None
    read_result: "ReadResult | None" = None

    def to_dict(self) -> dict:
        """Flatten the search result into the contract dict for REST/MCP."""
        out = {
            "success": self.success,
            "request_id": self.request_id,
            "memories": self.memories.to_dict(),
            "processing_time_ms": self.processing_time_ms,
        }
        if self.read_result is not None:
            out["read_result"] = self.read_result.to_dict()
        return out


@dataclass
class ReadResult:
    """Read-pipeline trace (debug / observability only)."""

    # 与 SearchResult.memories 相同的三路结果快照
    memories: SearchMemories
    # 融合打分后进入分组的条数
    final_count: int = 0
    # 读管线内部耗时（毫秒）
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        """Flatten the trace into the contract dict for pipeline_logs / API debug."""
        return {
            "memories": self.memories.to_dict(),
            "final_count": self.final_count,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class DigestResult:
    """Outcome of MemoryClient.digest: (dual) System2 cognition."""

    # digest 调用是否成功
    success: bool
    # 成功处理的 System2 app/user/agent scope 数（system1 恒为 0）
    processed: int = 0
    # 本次 digest 各阶段耗时与计数（性能分析用；非 dual 或无任务时为空）
    timing: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Flatten the digest result for the CLI/REST."""
        return asdict(self)


@dataclass
class DeleteResult:
    """Outcome of deleting a single memory by id."""

    # 是否删除成功
    success: bool
    # 错误码；成功时为 None
    error_code: int | None = None

    def to_dict(self) -> dict:
        """Flatten with the error_code only when present."""
        return _omit_none(asdict(self))


@dataclass
class DeleteBulkResult:
    """Outcome of delete_bulk for a user/app scope."""

    # 是否成功
    success: bool
    # 实际删除（软删）条数
    deleted: int = 0
    # 错误码；成功时为 None
    error_code: int | None = None

    def to_dict(self) -> dict:
        """Flatten with the error_code only when present."""
        return _omit_none(asdict(self))


@dataclass
class ScopeSummary:
    """One tenant scope and its memory count (from list_scopes)."""

    # 应用/产品命名空间
    app_id: str
    # 用户 id
    user_id: str
    # 可选 agent 维度；无则为 ""
    agent_id: str = ""
    # 该 scope 下记忆条数
    memory_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UpdateResult:
    """Outcome of MemoryClient.update (content replace + re-embed)."""

    # 是否更新成功
    success: bool
    # 被更新的节点 id；失败时可为 None
    memory_id: str | None = None
    # 错误码；成功时为 None
    error_code: int | None = None

    def to_dict(self) -> dict:
        """Flatten with the error_code only when present."""
        return _omit_none(asdict(self))
