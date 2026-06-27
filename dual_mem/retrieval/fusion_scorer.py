# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)

DEPRECATED (暂不删除)：本模块已被 ``retrieval/hybrid_engine.py`` 的池内 BM25 重排 + 图证据
融合取代，当前 hybrid / legacy 读路径都不再引用它，仅保留独立单测。后续清理再删。

@description: Fusion scorer for the hybrid read path. Combines per-anchor semantic similarity
with a multiplicative weight stack (time decay, access frequency, emotional arousal,
schema/long-tail boosts) and an additive RRF term that rewards documents surfaced by
multiple retrieval paths. Replaces the older intent-weighted RRF rerank.

W(d) = α · Sim × time_decay × log(1+freq) × arousal × SchemaBoost + rrf_weight · RRF
"""
import logging
import math
import time
from dataclasses import dataclass, field

from dual_mem.retrieval.anchor_search import AnchorNode
from dual_mem.storage.cache_store import CacheStore
from dual_mem.types import Layer, MemoryNode

logger = logging.getLogger("dual_mem.retrieval.fusion")


@dataclass
class FusionConfig:
    """Tunable knobs for the fusion W(d) score."""

    semantic_weight: float = 0.40        # α
    time_decay_lambda_per_day: float = 0.01
    arousal_beta: float = 0.30
    schema_boost: float = 1.20           # node activated by schema path
    longtail_boost: float = 1.50         # node tagged with LONG_TAIL meta
    rrf_k: int = 60
    rrf_weight: float = 0.15
    max_results: int = 30


@dataclass
class ScoredNode:
    """Per-node fusion breakdown (kept for debugging / pipeline trace)."""

    node: MemoryNode
    semantic_score: float = 0.0
    time_decay: float = 1.0
    frequency: float = 0.0
    arousal: float = 1.0
    schema_boost: float = 1.0
    rrf_score: float = 0.0
    final_score: float = 0.0
    source_paths: list[str] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return self.node.node_id


class FusionScorer:
    """Score and rank anchor candidates with the W(d) recency formula + RRF."""

    def __init__(
        self,
        *,
        config: FusionConfig | None = None,
        cache: CacheStore | None = None,
    ):
        self.config = config or FusionConfig()
        self.cache = cache

    def score_and_rank(
        self,
        *,
        anchors: list[AnchorNode],
        activated_schema_ids: set[str] | None = None,
        now_ts: int | None = None,
        max_results: int | None = None,
    ) -> list[ScoredNode]:
        """Score the deduped anchor list (multi-path), return ranked ScoredNodes.

        ``max_results`` overrides ``config.max_results`` for this call (the reader
        passes headroom derived from the request ``limit`` so a large top_k is filled).
        """
        if not anchors:
            return []
        activated = activated_schema_ids or set()
        now_ts = now_ts if now_ts is not None else int(time.time())

        # Build per-path rankings (rank 1 = top of that path) for the RRF term.
        path_rankings: dict[str, list[str]] = {}
        for anchor in anchors:
            path_rankings.setdefault(anchor.source_path, []).append(anchor.node_id)

        node_map: dict[str, ScoredNode] = {}
        for anchor in anchors:
            scored = node_map.get(anchor.node_id)
            if scored is None:
                scored = ScoredNode(
                    node=anchor.node,
                    semantic_score=anchor.score,
                    source_paths=[anchor.source_path],
                )
                node_map[anchor.node_id] = scored
            else:
                if anchor.score > scored.semantic_score:
                    scored.semantic_score = anchor.score
                if anchor.source_path not in scored.source_paths:
                    scored.source_paths.append(anchor.source_path)

        access_rows: dict[str, dict] = {}
        if self.cache is not None:
            try:
                access_rows = self.cache.get_access_batch(list(node_map.keys()))
            except Exception:
                access_rows = {}

        for nid, scored in node_map.items():
            scored.time_decay = self._time_decay(scored.node, now_ts)
            scored.frequency = self._frequency(nid, access_rows.get(nid))
            scored.arousal = self._arousal(scored.node)
            scored.schema_boost = self._schema_boost(scored.node, activated)
            scored.rrf_score = self._rrf(nid, path_rankings)
            scored.final_score = self._final(scored)

        ranked = sorted(node_map.values(), key=lambda s: s.final_score, reverse=True)
        logger.debug(
            "fusion anchors=%d unique=%d top_score=%.4f",
            len(anchors), len(node_map),
            ranked[0].final_score if ranked else 0.0,
        )
        cap = max_results if max_results and max_results > 0 else self.config.max_results
        return ranked[:cap]

    # ---- Per-dimension scorers ---------------------------------------------------------

    def _time_decay(self, node: MemoryNode, now_ts: int) -> float:
        """exp(-λ · Δdays); fresh nodes ≈ 1.0, year-old nodes decay smoothly."""
        ref = node.gmt_created or 0
        if ref <= 0:
            return 1.0
        delta_days = max(0.0, (now_ts - ref) / 86400.0)
        return math.exp(-self.config.time_decay_lambda_per_day * delta_days)

    def _frequency(self, node_id: str, access_row: dict | None = None) -> float:
        """log(1 + access_count) normalized to [0, 1] via log(101)."""
        row = access_row
        if row is None and self.cache is not None:
            try:
                row = self.cache.get_access(node_id)
            except Exception:
                return 0.0
        if not row:
            return 0.0
        count = int(row.get("access_count", 0) or 0)
        if count <= 0:
            return 0.0
        return min(1.0, math.log(1 + count) / math.log(101))

    def _arousal(self, node: MemoryNode) -> float:
        """1 + β·|arousal|; arousal sourced from node.custom written by extractor/reflector."""
        custom = node.custom or {}
        arousal = abs(float(custom.get("emotional_arousal", 0.0) or 0.0))
        return 1.0 + self.config.arousal_beta * arousal

    def _schema_boost(self, node: MemoryNode, activated_schema_ids: set[str]) -> float:
        """Schema-path or LONG_TAIL meta gives a multiplicative boost."""
        boost = 1.0
        if node.layer is Layer.L6_SCHEMA and node.node_id in activated_schema_ids:
            boost *= self.config.schema_boost
        custom = node.custom or {}
        meta_tags = custom.get("meta_tags") or []
        if isinstance(meta_tags, list) and "LONG_TAIL" in meta_tags:
            boost *= self.config.longtail_boost
        return boost

    def _rrf(self, node_id: str, path_rankings: dict[str, list[str]]) -> float:
        """Σ 1 / (k + rank_i(d)) over paths that surfaced this node."""
        k = self.config.rrf_k
        score = 0.0
        for ranking in path_rankings.values():
            try:
                rank = ranking.index(node_id) + 1
            except ValueError:
                continue
            score += 1.0 / (k + rank)
        return score

    def _final(self, scored: ScoredNode) -> float:
        """Multiplicative base (semantic × decay × freq × arousal × schema) + additive RRF."""
        cfg = self.config
        base = (
            cfg.semantic_weight
            * scored.semantic_score
            * scored.time_decay
            * max(0.1, scored.frequency + 0.5)  # avoid freq=0 zeroing the product
            * scored.arousal
            * scored.schema_boost
        )
        return base + cfg.rrf_weight * scored.rrf_score
