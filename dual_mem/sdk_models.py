# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SDK-level dataclass models returned by MemoryClient. The SDK keeps strongly
typed objects internally (IDE-friendly attribute access, refactor-safe) and only flattens to
plain dicts at REST/MCP/CLI boundaries via .to_dict().
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _omit_none(data: dict) -> dict:
    """Drop keys whose value is None, leaving empty lists/dicts untouched."""
    return {k: v for k, v in data.items() if v is not None}


@dataclass
class ChatMessage:
    """One turn of a multi-turn write input. role: user / assistant / system."""

    role: str
    content: str

    def to_dict(self) -> dict:
        """Plain dict form so callers can treat ChatMessage and {role,content} dicts identically."""
        return {"role": self.role, "content": self.content}


@dataclass
class GateResult:
    """Outcome of the attentional gate: pass/reject plus the three component scores."""

    passed: bool
    gate_score: float
    novelty: float
    biographical_relevance: float
    emotional_arousal: float
    reason: str = ""
    scoring_method: str = "heuristic"
    top_similar_id: str | None = None
    top_similar_score: float = 0.0

    def to_dict(self) -> dict:
        """Flatten this gate result into a dict for logging or contract responses."""
        return _omit_none(asdict(self))


@dataclass
class WriteResult:
    """Outcome of a single MemoryClient.add call (raw id + processing metadata)."""

    success: bool
    memory_id: str
    request_id: str
    processing_time_ms: float
    gate_passed: bool = True
    gate_score: float | None = None
    extracted_count: int = 0
    extra_node_ids: list[str] = field(default_factory=list)
    is_ephemeral: bool = False
    error_code: int | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Flatten the write result into the contract dict (None fields omitted)."""
        return _omit_none(asdict(self))


@dataclass
class EvolutionItem:
    """One historical version inside a memory's evolution chain (newest -> oldest)."""

    node_id: str
    content: str
    layer: str
    memory_at: int | None = None
    gmt_created: int | None = None
    speculate: str | None = None

    def to_dict(self) -> dict:
        """Serialize a single chain entry as a dict, omitting null fields."""
        return _omit_none(asdict(self))


@dataclass
class MemoryItem:
    """One scored memory record returned from the read path."""

    memory_id: str
    content: str
    category: str
    score: float
    tags: list[str] = field(default_factory=list)
    memory_at: int | None = None
    gmt_created: int | None = None
    gmt_modified: int | None = None
    evolution_chain: list[EvolutionItem] | None = None

    def to_dict(self) -> dict:
        """Serialize this memory item with its (optional) evolution chain expanded."""
        out: dict[str, Any] = {
            "memory_id": self.memory_id,
            "content": self.content,
            "category": self.category,
            "score": round(self.score, 4),
            "tags": list(self.tags),
        }
        if self.memory_at is not None:
            out["memory_at"] = self.memory_at
        if self.gmt_created is not None:
            out["gmt_created"] = self.gmt_created
        if self.gmt_modified is not None:
            out["gmt_modified"] = self.gmt_modified
        if self.evolution_chain:
            out["evolution_chain"] = [item.to_dict() for item in self.evolution_chain]
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
        }


@dataclass
class SearchMemories:
    """The three routing groups returned by the reader."""

    profile: list[MemoryItem] = field(default_factory=list)
    proactive: list[MemoryItem] = field(default_factory=list)
    normal: list[MemoryItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Flatten each route's items into plain dicts."""
        return {
            "profile": [item.to_dict() for item in self.profile],
            "proactive": [item.to_dict() for item in self.proactive],
            "normal": [item.to_dict() for item in self.normal],
        }

    def flatten(self, *, limit: int | None = None) -> list[MemoryItem]:
        """Merge profile/proactive/normal and sort by score descending."""
        items = [*self.profile, *self.proactive, *self.normal]
        items.sort(key=lambda x: x.score, reverse=True)
        if limit is not None and limit >= 0:
            return items[:limit]
        return items

    def to_search_results(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Current-state search hits for QA pipelines (see ``MemoryItem.to_search_result``)."""
        return [item.to_search_result() for item in self.flatten(limit=limit)]


@dataclass
class SearchResult:
    """Outcome of MemoryClient.search: grouped memories + processing metadata.

    ``read_result`` is populated only when the caller passes ``debug=True``; it carries the
    per-stage read-pipeline trace (intent, anchor path counts, expansion edges, fusion
    final count, elapsed) for observability — never required for normal use."""

    success: bool
    request_id: str
    memories: SearchMemories
    processing_time_ms: float
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
    """Internal read-pipeline trace (debug / observability). Reader returns SearchMemories
    on the contract surface; ReadResult is for callers that want to inspect anchor/expansion
    decisions made during a search."""

    memories: SearchMemories
    intent: str = ""
    target_layers: list[str] = field(default_factory=list)
    has_temporal: bool = False
    anchor_path_counts: dict[str, int] = field(default_factory=dict)
    anchor_count: int = 0
    expanded_count: int = 0
    edge_counts: dict[str, int] = field(default_factory=dict)
    final_count: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        """Flatten the trace into the contract dict for pipeline_logs / API debug."""
        return {
            "memories": self.memories.to_dict(),
            "intent": self.intent,
            "target_layers": list(self.target_layers),
            "has_temporal": self.has_temporal,
            "anchor_path_counts": dict(self.anchor_path_counts),
            "anchor_count": self.anchor_count,
            "expanded_count": self.expanded_count,
            "edge_counts": dict(self.edge_counts),
            "final_count": self.final_count,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


@dataclass
class DigestResult:
    """Outcome of MemoryClient.digest: drained tasks plus cross-domain core schema count."""

    success: bool
    processed: int = 0
    cores_created: int = 0

    def to_dict(self) -> dict:
        """Flatten the digest result for the CLI/REST."""
        return asdict(self)


@dataclass
class DeleteResult:
    """Outcome of a single delete: success flag plus optional error code."""

    success: bool
    error_code: int | None = None

    def to_dict(self) -> dict:
        """Flatten with the error_code only when present."""
        return _omit_none(asdict(self))


@dataclass
class DeleteBulkResult:
    """Outcome of a bulk delete: how many memories were removed in scope."""

    success: bool
    deleted: int = 0
    error_code: int | None = None

    def to_dict(self) -> dict:
        """Flatten with the error_code only when present."""
        return _omit_none(asdict(self))


@dataclass
class ScopeSummary:
    """One tenant scope (app + user + optional agent) and how many memories it holds."""

    app_id: str
    user_id: str
    agent_id: str = ""
    memory_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UpdateResult:
    """Outcome of an update call: success flag and the updated memory id."""

    success: bool
    memory_id: str | None = None
    error_code: int | None = None

    def to_dict(self) -> dict:
        """Flatten with the error_code only when present."""
        return _omit_none(asdict(self))
