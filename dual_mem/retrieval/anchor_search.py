# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)

DEPRECATED (暂不删除)：本模块已被 ``retrieval/hybrid_engine.py`` 的并行召回（语义 / L0 profile /
L6 graph schema）取代，当前 hybrid / legacy 读路径都不再引用它，仅保留独立单测。后续清理再删。

@description: Multi-path anchor search for the hybrid read flow. Runs the configured retrieval
paths (semantic / entity / temporal / schema / intention) in parallel and merges results
into a deduped anchor list with per-path counts. Each path is independent and tolerates
storage failures so the rest of the request still produces an answer.
"""
import asyncio
import logging
import math
from dataclasses import dataclass, field

from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval import bm25
from dual_mem.retrieval.query_understanding import QueryUnderstanding
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.retrieval.anchor")


# Retrieval paths
PATH_SEMANTIC = "semantic"
PATH_ENTITY = "entity"
PATH_TEMPORAL = "temporal"
PATH_SCHEMA = "schema"
PATH_INTENTION = "intention"

_DEFAULT_LAYERS = [
    Layer.L2_FACT,
    Layer.L3_SUMMARY,
    Layer.L4_IDENTITY,
    Layer.L5_KNOWLEDGE,
    Layer.L1_RAW,
]


@dataclass
class AnchorNode:
    """One retrieval anchor: the node that triggered it plus which path found it."""

    node: MemoryNode
    score: float
    source_path: str

    @property
    def node_id(self) -> str:
        return self.node.node_id


@dataclass
class AnchorSearchResult:
    """Output of multi-path anchor search: deduped anchors + per-path counts."""

    anchors: list[AnchorNode] = field(default_factory=list)
    path_counts: dict[str, int] = field(default_factory=dict)
    activated_schemas: list[MemoryNode] = field(default_factory=list)
    triggered_intentions: list[MemoryNode] = field(default_factory=list)


class AnchorSearchEngine:
    """Multi-path anchor search engine (parallel paths, merged anchor list)."""

    def __init__(
        self,
        *,
        factory: ComponentFactory,
        semantic_limit: int = 15,
        temporal_limit: int = 20,
        entity_limit: int = 20,
        schema_limit: int = 8,
        schema_threshold: float = 0.3,
        intention_threshold: float = 0.4,
        semantic_threshold: float = 0.3,
        entity_pool_limit: int = 200,
    ):
        self.factory = factory
        self.semantic_limit = semantic_limit
        self.temporal_limit = temporal_limit
        self.entity_limit = entity_limit
        self.schema_limit = schema_limit
        self.schema_threshold = schema_threshold
        self.intention_threshold = intention_threshold
        self.semantic_threshold = semantic_threshold
        self.entity_pool_limit = entity_pool_limit

    async def search(
        self,
        *,
        query: str,  # noqa: ARG002 — kept for API symmetry; entity path uses understanding.keywords
        query_embedding: list[float],
        understanding: QueryUnderstanding,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None = None,
        session_ids: list[str] | None = None,
        layers: list[Layer] | None = None,
        created_after: int | None = None,
        intention_enabled: bool = False,
        recall_limit: int | None = None,
    ) -> AnchorSearchResult:
        """Run all enabled paths concurrently; return merged anchors + per-path counts.

        ``recall_limit`` lets the caller (the reader's ``limit``) widen the per-path
        candidate pools so a large ``top_k`` is actually filled. The fixed defaults
        (semantic_limit/temporal_limit) are treated as a floor.
        """
        target_layers = layers or understanding.target_layers or _DEFAULT_LAYERS
        sem_limit = self.semantic_limit
        temp_limit = self.temporal_limit
        ent_limit = self.entity_limit
        if recall_limit and recall_limit > 0:
            sem_limit = max(sem_limit, recall_limit)
            temp_limit = max(temp_limit, recall_limit)
            ent_limit = max(ent_limit, recall_limit)

        tasks: dict = {
            PATH_SEMANTIC: self._semantic(
                query_embedding=query_embedding,
                app_ids=app_ids,
                user_id=user_id,
                agent_ids=agent_ids,
                session_ids=session_ids,
                layers=target_layers,
                created_after=created_after,
                limit=sem_limit,
            ),
            PATH_SCHEMA: self._schema(
                query_embedding=query_embedding,
                app_ids=app_ids,
                user_id=user_id,
            ),
        }
        if understanding.has_temporal or created_after is not None:
            tasks[PATH_TEMPORAL] = self._temporal(
                app_ids=app_ids,
                user_id=user_id,
                agent_ids=agent_ids,
                session_ids=session_ids,
                layers=target_layers,
                created_after=created_after if created_after is not None else understanding.time_from,
                limit=temp_limit,
            )
        if understanding.keywords:
            tasks[PATH_ENTITY] = self._entity(
                keywords=understanding.keywords,
                app_ids=app_ids,
                user_id=user_id,
                agent_ids=agent_ids,
                session_ids=session_ids,
                layers=target_layers,
                limit=ent_limit,
            )
        if intention_enabled:
            tasks[PATH_INTENTION] = self._intention(
                query_embedding=query_embedding,
                app_ids=app_ids,
                user_id=user_id,
            )

        keys = list(tasks.keys())
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        out = AnchorSearchResult()
        for key, res in zip(keys, results):
            if isinstance(res, BaseException):
                out.path_counts[key] = 0
                logger.warning("anchor path %s failed: %s", key, res)
                continue
            anchors: list[AnchorNode] = res
            if key == PATH_SCHEMA:
                out.activated_schemas = [a.node for a in anchors]
            if key == PATH_INTENTION:
                out.triggered_intentions = [a.node for a in anchors]
            self._merge(out.anchors, anchors)
            out.path_counts[key] = len(anchors)
        logger.debug(
            "anchor user=%s paths=%s total=%d", user_id, dict(out.path_counts), len(out.anchors)
        )
        return out

    # ---- Paths -------------------------------------------------------------------------

    async def _semantic(
        self,
        *,
        query_embedding: list[float],
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None,
        session_ids: list[str] | None,
        layers: list[Layer],
        created_after: int | None,
        limit: int,
    ) -> list[AnchorNode]:
        """Path 1: vector semantic neighbors over the target layers."""
        where = build_filter(
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            layers=layers,
            statuses=[MemoryStatus.ACTIVE],
            created_after=created_after,
        )
        nodes = await asyncio.to_thread(
            self.factory.vector.query,
            embedding=query_embedding,
            where=where,
            top_k=math.ceil(limit * 1.5),
        )
        return [
            AnchorNode(node=n, score=n.score, source_path=PATH_SEMANTIC)
            for n in nodes
            if n.score >= self.semantic_threshold
        ][:limit]

    async def _entity(
        self,
        *,
        keywords: list[str],
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None,
        session_ids: list[str] | None,
        layers: list[Layer],
        limit: int,
    ) -> list[AnchorNode]:
        """Path 2: BM25 keyword scoring over a bounded candidate pool (replaces substring scan)."""
        where = build_filter(
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            layers=layers,
            statuses=[MemoryStatus.ACTIVE],
        )
        pool = await asyncio.to_thread(
            self.factory.vector.get_many,
            where,
            self.entity_pool_limit,
        )
        if not pool:
            return []

        ranked = bm25.score_and_rank(
            keywords,
            [(n.node_id, n.content) for n in pool],
        )
        node_by_id = {n.node_id: n for n in pool}
        out: list[AnchorNode] = []
        for nid, bm25_norm in ranked:
            if bm25_norm <= 0:
                continue
            node = node_by_id.get(nid)
            if node is None:
                continue
            # bm25_norm is already max-normalized 0–1; map to semantic-comparable range
            # without clamping every hit to ~0.9 (which would dominate fusion max()).
            score = 0.25 + 0.45 * bm25_norm
            out.append(AnchorNode(node=node, score=score, source_path=PATH_ENTITY))
        return out[:limit]

    async def _temporal(
        self,
        *,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None,
        session_ids: list[str] | None,
        layers: list[Layer],
        created_after: int | None,
        limit: int,
    ) -> list[AnchorNode]:
        """Path 3: time-window lookup; surfaces nodes created after the parsed cutoff."""
        if created_after is None:
            return []
        where = build_filter(
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            layers=layers,
            statuses=[MemoryStatus.ACTIVE],
            created_after=created_after,
        )
        nodes = await asyncio.to_thread(
            self.factory.vector.get_many,
            where,
            limit,
        )
        nodes.sort(key=lambda n: n.gmt_created or 0, reverse=True)
        return [
            AnchorNode(node=n, score=0.5, source_path=PATH_TEMPORAL)
            for n in nodes[:limit]
        ]

    async def _schema(
        self,
        *,
        query_embedding: list[float],
        app_ids: list[str],
        user_id: str,
    ) -> list[AnchorNode]:
        """Path 4: schema match in the graph store; dual-only — empty when graph is None."""
        graph = self.factory.graph
        if graph is None:
            return []
        try:
            hits = await asyncio.to_thread(
                graph.query_by_embedding,
                layer=Layer.L6_SCHEMA.value,
                user_id=user_id,
                app_ids=app_ids,
                embedding=query_embedding,
                top_k=self.schema_limit,
            )
        except Exception:
            return []
        out: list[AnchorNode] = []
        for g in hits:
            if g.score < self.schema_threshold:
                continue
            node = _graph_to_node(g)
            out.append(AnchorNode(node=node, score=g.score, source_path=PATH_SCHEMA))
        return out

    async def _intention(
        self,
        *,
        query_embedding: list[float],
        app_ids: list[str],
        user_id: str,
    ) -> list[AnchorNode]:
        """Path 5: intention check in the graph store; dual-only with intention_limit>0."""
        graph = self.factory.graph
        if graph is None:
            return []
        try:
            hits = await asyncio.to_thread(
                graph.query_by_embedding,
                layer=Layer.L7_INTENTION.value,
                user_id=user_id,
                app_ids=app_ids,
                embedding=query_embedding,
                top_k=self.schema_limit,
            )
        except Exception:
            return []
        out: list[AnchorNode] = []
        for g in hits:
            if g.score < self.intention_threshold:
                continue
            node = _graph_to_node(g)
            out.append(AnchorNode(node=node, score=g.score, source_path=PATH_INTENTION))
        return out

    @staticmethod
    def _merge(existing: list[AnchorNode], new_anchors: list[AnchorNode]) -> None:
        """Merge new anchors into existing, keeping the higher score per node id."""
        index = {a.node_id: i for i, a in enumerate(existing)}
        for anchor in new_anchors:
            if anchor.node_id in index:
                idx = index[anchor.node_id]
                if anchor.score > existing[idx].score:
                    existing[idx] = anchor
            else:
                index[anchor.node_id] = len(existing)
                existing.append(anchor)


def _graph_to_node(g) -> MemoryNode:
    """Adapt a GraphNode into a MemoryNode so all paths return uniform types."""
    node = MemoryNode(
        content=g.content,
        layer=Layer(g.layer),
        app_id=g.app_id,
        user_id=g.user_id,
        agent_id=g.agent_id,
        tags=list(g.tags),
        node_id=g.node_id,
        gmt_created=g.gmt_created,
    )
    node.score = g.score
    return node
