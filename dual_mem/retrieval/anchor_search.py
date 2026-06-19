# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Multi-path anchor search for the V2 read flow. Runs the configured retrieval
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
        schema_limit: int = 8,
        schema_threshold: float = 0.3,
        intention_threshold: float = 0.4,
        semantic_threshold: float = 0.3,
    ):
        self.factory = factory
        self.semantic_limit = semantic_limit
        self.temporal_limit = temporal_limit
        self.schema_limit = schema_limit
        self.schema_threshold = schema_threshold
        self.intention_threshold = intention_threshold
        self.semantic_threshold = semantic_threshold

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
    ) -> AnchorSearchResult:
        """Run all enabled paths concurrently; return merged anchors + per-path counts."""
        target_layers = layers or understanding.target_layers or _DEFAULT_LAYERS

        tasks: dict = {
            PATH_SEMANTIC: self._semantic(
                query_embedding=query_embedding,
                app_ids=app_ids,
                user_id=user_id,
                agent_ids=agent_ids,
                session_ids=session_ids,
                layers=target_layers,
                created_after=created_after,
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
            )
        if understanding.keywords:
            tasks[PATH_ENTITY] = self._entity(
                keywords=understanding.keywords,
                app_ids=app_ids,
                user_id=user_id,
                agent_ids=agent_ids,
                session_ids=session_ids,
                layers=target_layers,
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
        nodes = self.factory.vector.query(
            embedding=query_embedding,
            where=where,
            top_k=math.ceil(self.semantic_limit * 1.5),
        )
        return [
            AnchorNode(node=n, score=n.score, source_path=PATH_SEMANTIC)
            for n in nodes
            if n.score >= self.semantic_threshold
        ][: self.semantic_limit]

    async def _entity(
        self,
        *,
        keywords: list[str],
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None,
        session_ids: list[str] | None,
        layers: list[Layer],
    ) -> list[AnchorNode]:
        """Path 2: keyword-substring lookup over content (zero-LLM, complements semantic)."""
        where = build_filter(
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            layers=layers,
            statuses=[MemoryStatus.ACTIVE],
        )
        # Pull a candidate pool then intersect by keyword.
        pool = self.factory.vector.get_many(where, limit=200)
        out: list[AnchorNode] = []
        for node in pool:
            content_lower = node.content.lower()
            hits = sum(
                1
                for kw in keywords
                if kw.lower() in content_lower or kw in node.content
            )
            if hits == 0:
                continue
            # Score by hit ratio; bounded to [0.3, 0.9].
            score = min(0.9, 0.3 + 0.15 * hits)
            out.append(AnchorNode(node=node, score=score, source_path=PATH_ENTITY))
        out.sort(key=lambda a: a.score, reverse=True)
        return out[:20]

    async def _temporal(
        self,
        *,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None,
        session_ids: list[str] | None,
        layers: list[Layer],
        created_after: int | None,
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
        nodes = self.factory.vector.get_many(where, limit=self.temporal_limit)
        # Stable temporal ordering (newest first).
        nodes.sort(key=lambda n: n.gmt_created or 0, reverse=True)
        return [
            AnchorNode(node=n, score=0.5, source_path=PATH_TEMPORAL)
            for n in nodes[: self.temporal_limit]
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
            hits = graph.query_by_embedding(
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
            hits = graph.query_by_embedding(
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
