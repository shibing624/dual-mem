# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async Reader. Default ``reader_mode=hybrid`` runs anchor + fusion pipeline
(QueryUnderstanding → AnchorSearch 5 paths → GraphExpander 1-hop → FusionScorer),
then splits into profile/proactive/normal routes and walks the evolution chain.
No LLM on read; query embedding + heuristic QU only. ``legacy`` keeps three-route
vector recall + BM25/RRF rerank on the normal route.
"""
import asyncio
import logging
import math
import time

from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval import bm25, rrf
from dual_mem.retrieval.anchor_search import AnchorNode, AnchorSearchEngine
from dual_mem.retrieval.evolution import expand_evolution_chains
from dual_mem.retrieval.fusion_scorer import FusionScorer
from dual_mem.retrieval.graph_expander import GraphExpander
from dual_mem.retrieval.intent import (
    INTENT_WEIGHTS_2CHANNEL,
    classify_intent,
    extract_keywords,
)
from dual_mem.retrieval.query_understanding import understand
from dual_mem.retrieval.reconsolidation import ReconsolidationHook
from dual_mem.sdk_models import EvolutionItem, MemoryItem, ReadResult, SearchMemories
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.retrieval.reader")

PROFILE_LAYERS = [Layer.L0_BASIC_INFO, Layer.L4_IDENTITY]
PROACTIVE_LAYERS = [Layer.L7_INTENTION]
NORMAL_LAYERS = [Layer.L2_FACT, Layer.L5_KNOWLEDGE, Layer.L3_SUMMARY, Layer.L1_RAW]

_OVERFETCH = 1.5
_PROFILE_FULL = 100

_IDENTITY_VALS = {Layer.L0_BASIC_INFO.value, Layer.L4_IDENTITY.value}
_SCHEMA_VALS = {Layer.L6_SCHEMA.value}
_PROFILE_VALS = _IDENTITY_VALS | _SCHEMA_VALS
_PROACTIVE_VALS = {Layer.L7_INTENTION.value}

# Default vector-store layer set when QueryUnderstanding has no opinion. L6 is excluded
# here because schemas are queried via the graph path; L7 is excluded because the proactive
# route is opt-in via intention_limit>0 (and also goes through graph).
_DEFAULT_VDB_LAYERS = [
    Layer.L0_BASIC_INFO,
    Layer.L1_RAW,
    Layer.L2_FACT,
    Layer.L3_SUMMARY,
    Layer.L4_IDENTITY,
    Layer.L5_KNOWLEDGE,
]


def _profile_quota_select(
    items: list[dict],
    total_limit: int,
    identity_vals: set[str],
    schema_vals: set[str],
) -> list[dict]:
    """Select profile items by quota: 40% identity, 40% schema, 20% free competition."""
    if total_limit <= 0 or not items:
        return items

    id_pool: list[dict] = []
    sc_pool: list[dict] = []
    for it in items:
        lv = it["node"].layer.value
        if lv in identity_vals:
            id_pool.append(it)
        elif lv in schema_vals:
            sc_pool.append(it)
        else:
            id_pool.append(it)

    id_quota = max(1, int(total_limit * 0.4))
    sc_quota = max(1, int(total_limit * 0.4))
    id_take = id_pool[:id_quota]
    sc_take = sc_pool[:sc_quota]

    free_slots = total_limit - len(id_take) - len(sc_take)
    if free_slots > 0:
        free_pool = sorted(
            id_pool[id_quota:] + sc_pool[sc_quota:],
            key=lambda x: x["score"],
            reverse=True,
        )
        free_take = free_pool[:free_slots]
    else:
        free_take = []

    return sorted(
        id_take + sc_take + free_take,
        key=lambda x: x["score"],
        reverse=True,
    )


class Reader:
    """Read path: V2 hybrid (default) or legacy three-route, both returning grouped MemoryItems."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory
        self.reconsolidation = ReconsolidationHook(factory=factory)
        self.anchor_engine = AnchorSearchEngine(factory=factory)
        self.expander = GraphExpander(factory=factory)
        self.fusion = FusionScorer(cache=factory.cache)
        # Handle to the most recent fire-and-forget reconsolidation task. Callers (e.g.
        # MemoryClient.search in per_write mode) can await this BEFORE draining the
        # reconsolidation queue, so the drain never races ahead of its own enqueue.
        self.last_reconsolidation_task: asyncio.Task | None = None

    async def search(
        self,
        *,
        query: str,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None = None,
        session_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        created_after: int | None = None,
        request_id: str | None = None,
    ) -> SearchMemories:
        """Recall, expand and rerank memories, returning profile/proactive/normal groups."""
        memories, _ = await self._search(
            query=query, app_ids=app_ids, user_id=user_id,
            agent_ids=agent_ids, session_ids=session_ids,
            limit=limit, min_score=min_score,
            profile_limit=profile_limit, profile_min_score=profile_min_score,
            intention_limit=intention_limit, created_after=created_after,
            request_id=request_id, collect_trace=False,
        )
        return memories

    async def search_with_trace(
        self,
        *,
        query: str,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None = None,
        session_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        created_after: int | None = None,
        request_id: str | None = None,
    ) -> tuple[SearchMemories, ReadResult]:
        """Same as ``search`` but also returns a populated ``ReadResult`` for debugging.

        ReadResult exposes the per-stage counters (anchor path counts, expansion edges,
        fusion final count, evolution chain count, elapsed_ms) so callers can render or log
        the read pipeline without re-querying the store. Used by ``MemoryClient.search(debug=True)``.
        """
        return await self._search(
            query=query, app_ids=app_ids, user_id=user_id,
            agent_ids=agent_ids, session_ids=session_ids,
            limit=limit, min_score=min_score,
            profile_limit=profile_limit, profile_min_score=profile_min_score,
            intention_limit=intention_limit, created_after=created_after,
            request_id=request_id, collect_trace=True,
        )

    async def _search(
        self,
        *,
        query: str,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None,
        session_ids: list[str] | None,
        limit: int,
        min_score: float,
        profile_limit: int,
        profile_min_score: float,
        intention_limit: int,
        created_after: int | None,
        request_id: str | None,
        collect_trace: bool,
    ) -> tuple[SearchMemories, ReadResult]:
        """Internal: shared body of ``search`` / ``search_with_trace``."""
        mode = self.factory.settings.reader_mode
        start = time.perf_counter()
        rid = request_id or "search"
        trace = ReadResult(memories=SearchMemories()) if collect_trace else None

        # Pipeline trace: query understanding (cheap, always logged for traceability).
        try:
            u = understand(query)
            if trace is not None:
                trace.intent = u.intent
                trace.target_layers = [layer.value for layer in u.target_layers]
                trace.has_temporal = u.has_temporal
            self.factory.cache.log_pipeline(
                request_id=rid,
                stage="READ_QU",
                payload={
                    "intent": u.intent,
                    "target_layers": [layer.value for layer in u.target_layers],
                    "has_temporal": u.has_temporal,
                },
            )
        except Exception:
            pass

        if mode == "legacy":
            memories = await self._search_legacy(
                query=query,
                app_ids=app_ids,
                user_id=user_id,
                agent_ids=agent_ids,
                session_ids=session_ids,
                limit=limit,
                min_score=min_score,
                profile_limit=profile_limit,
                profile_min_score=profile_min_score,
                intention_limit=intention_limit,
                created_after=created_after,
            )
        else:
            memories = await self._search_hybrid(
                query=query,
                app_ids=app_ids,
                user_id=user_id,
                agent_ids=agent_ids,
                session_ids=session_ids,
                limit=limit,
                min_score=min_score,
                profile_limit=profile_limit,
                profile_min_score=profile_min_score,
                intention_limit=intention_limit,
                created_after=created_after,
                request_id=rid,
                trace=trace,
            )

        # Reconsolidation Hook (fire-and-forget; never blocks the response).
        recalled = {
            "profile": [m.memory_id for m in memories.profile],
            "proactive": [m.memory_id for m in memories.proactive],
            "normal": [m.memory_id for m in memories.normal],
        }
        task = asyncio.create_task(
            self.reconsolidation.process(
                query=query,
                recalled_by_route=recalled,
                user_id=user_id,
                app_id=app_ids[0] if app_ids else "",
                agent_id="",
            )
        )
        task.add_done_callback(_swallow)
        self.last_reconsolidation_task = task

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if trace is not None:
            trace.memories = memories
            trace.elapsed_ms = elapsed_ms
            trace.final_count = (
                len(memories.profile) + len(memories.proactive) + len(memories.normal)
            )
        return memories, trace if trace is not None else ReadResult(memories=memories, elapsed_ms=elapsed_ms)

    # ---- Hybrid (V2) ---------------------------------------------------------------

    async def _search_hybrid(
        self,
        *,
        query: str,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None,
        session_ids: list[str] | None,
        limit: int,
        min_score: float,
        profile_limit: int,
        profile_min_score: float,
        intention_limit: int,
        created_after: int | None,
        request_id: str = "search",
        trace: ReadResult | None = None,
    ) -> SearchMemories:
        """V2 read flow: QU → Anchor 5 paths → GraphExpander → Fusion → split into routes."""
        understanding = understand(query)
        if created_after is None and understanding.time_from is not None:
            created_after = understanding.time_from

        embedding = await self.factory.embed.embed(query)

        # Multi-path anchors. QU-suggested layers PLUS the always-on profile layers
        # (L0/L4) — profile content must be reachable on every query, intent classification
        # is just a hint. The schema path queries the graph store separately.
        suggested = list(understanding.target_layers) if understanding.target_layers else []
        target_layers: list[Layer] = []
        seen_layers: set[Layer] = set()
        for layer in [*suggested, *_DEFAULT_VDB_LAYERS]:
            if layer not in seen_layers:
                seen_layers.add(layer)
                target_layers.append(layer)
        anchor_result = await self.anchor_engine.search(
            query=query,
            query_embedding=embedding,
            understanding=understanding,
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            layers=target_layers,
            created_after=created_after,
            intention_enabled=intention_limit > 0,
        )
        if trace is not None:
            trace.anchor_path_counts = dict(anchor_result.path_counts)
            trace.anchor_count = len(anchor_result.anchors)
        try:
            self.factory.cache.log_pipeline(
                request_id=request_id,
                stage="READ_ANCHOR",
                payload={
                    "path_counts": dict(anchor_result.path_counts),
                    "anchor_count": len(anchor_result.anchors),
                },
            )
        except Exception:
            pass

        # 1-hop graph expansion from the merged anchor list.
        expansion = self.expander.expand(
            anchors=anchor_result.anchors, app_ids=app_ids, user_id=user_id
        )
        if trace is not None:
            trace.expanded_count = len(expansion.expanded)
            trace.edge_counts = dict(expansion.edge_counts)
        try:
            self.factory.cache.log_pipeline(
                request_id=request_id,
                stage="READ_EXPAND",
                payload={
                    "expanded_count": len(expansion.expanded),
                    "edge_counts": dict(expansion.edge_counts),
                },
            )
        except Exception:
            pass

        all_anchors: list[AnchorNode] = list(anchor_result.anchors) + list(expansion.expanded)
        activated_schema_ids = {s.node_id for s in anchor_result.activated_schemas}
        scored = self.fusion.score_and_rank(
            anchors=all_anchors, activated_schema_ids=activated_schema_ids
        )
        try:
            self.factory.cache.log_pipeline(
                request_id=request_id,
                stage="READ_FUSION",
                payload={
                    "scored_count": len(scored),
                    "top_score": round(scored[0].final_score, 4) if scored else 0.0,
                },
            )
        except Exception:
            pass

        # Map ScoredNode → internal item dict so evolution_chain expansion stays uniform.
        items: list[dict] = [
            {"node": s.node, "score": s.final_score} for s in scored
        ]
        deduped = expand_evolution_chains(vector=self.factory.vector, hits=[it["node"] for it in items])
        try:
            chain_count = sum(1 for it in deduped if it.get("evolution_chain"))
            self.factory.cache.log_pipeline(
                request_id=request_id,
                stage="READ_EVOLUTION",
                payload={
                    "deduped_count": len(deduped),
                    "with_chain_count": chain_count,
                },
            )
        except Exception:
            pass
        # expand_evolution_chains returns full item dicts; align ordering with the original `items`.
        deduped_by_id = {it["node"].node_id: it for it in deduped}
        ordered: list[dict] = []
        seen: set[str] = set()
        for it in items:
            nid = it["node"].node_id
            if nid in seen:
                continue
            seen.add(nid)
            entry = deduped_by_id.get(nid, it)
            entry["score"] = max(entry.get("score", 0.0), it["score"])
            ordered.append(entry)

        # Split into routes by layer.
        profile_items: list[dict] = []
        proactive_items: list[dict] = []
        normal_items: list[dict] = []
        for it in ordered:
            lv = it["node"].layer.value
            if lv in _PROFILE_VALS:
                profile_items.append(it)
            elif lv in _PROACTIVE_VALS:
                proactive_items.append(it)
            else:
                normal_items.append(it)

        # Per-route trimming (mirror legacy thresholds for contract compatibility).
        effective_profile_limit = profile_limit if profile_limit > 0 else _PROFILE_FULL
        profile_items = [it for it in profile_items if it["score"] >= profile_min_score]
        profile_items = _profile_quota_select(
            profile_items, effective_profile_limit, _IDENTITY_VALS, _SCHEMA_VALS
        )
        normal_items = [it for it in normal_items if it["score"] >= min_score][:limit]
        proactive_items = (
            [it for it in proactive_items if it["score"] >= min_score][:intention_limit]
            if intention_limit > 0
            else []
        )

        logger.info(
            "read_hybrid user=%s anchors=%d expanded=%d scored=%d profile=%d normal=%d proactive=%d",
            user_id,
            len(anchor_result.anchors),
            len(expansion.expanded),
            len(scored),
            len(profile_items),
            len(normal_items),
            len(proactive_items),
        )

        return SearchMemories(
            profile=[self._item_to_memory(it) for it in profile_items],
            proactive=[self._item_to_memory(it) for it in proactive_items],
            normal=[self._item_to_memory(it) for it in normal_items],
        )

    # ---- Legacy (BM25+RRF baseline) -------------------------------------------

    async def _search_legacy(
        self,
        *,
        query: str,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None,
        session_ids: list[str] | None,
        limit: int,
        min_score: float,
        profile_limit: int,
        profile_min_score: float,
        intention_limit: int,
        created_after: int | None,
    ) -> SearchMemories:
        """Original three-route + BM25/RRF rerank pipeline (kept as horizontal-eval baseline)."""
        understanding = understand(query)
        if created_after is None and understanding.time_from is not None:
            created_after = understanding.time_from

        embedding = await self.factory.embed.embed(query)
        vector = self.factory.vector

        effective_profile_limit = profile_limit if profile_limit > 0 else _PROFILE_FULL

        profile_where = build_filter(
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            layers=PROFILE_LAYERS,
            statuses=[MemoryStatus.ACTIVE],
            created_after=created_after,
        )
        profile_nodes = [
            n
            for n in vector.query(
                embedding=embedding,
                where=profile_where,
                top_k=math.ceil(effective_profile_limit * _OVERFETCH),
            )
            if n.score >= profile_min_score
        ]

        normal_where = build_filter(
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            layers=NORMAL_LAYERS,
            statuses=[MemoryStatus.ACTIVE],
            created_after=created_after,
        )
        normal_nodes = [
            n
            for n in vector.query(
                embedding=embedding,
                where=normal_where,
                top_k=math.ceil(limit * _OVERFETCH),
            )
            if n.score >= min_score
        ]

        graph = self.factory.graph
        proactive_nodes: list[MemoryNode] = []
        graph_schema_hits: list[MemoryNode] = []
        if graph is not None:
            graph_schema_hits = [
                self._graph_to_node(g)
                for g in graph.query_by_embedding(
                    layer=Layer.L6_SCHEMA.value,
                    user_id=user_id,
                    app_ids=app_ids,
                    embedding=embedding,
                    top_k=math.ceil(effective_profile_limit * _OVERFETCH),
                )
                if g.score >= profile_min_score
            ]
            profile_nodes += graph_schema_hits
            if intention_limit > 0:
                proactive_nodes = [
                    self._graph_to_node(g)
                    for g in graph.query_by_embedding(
                        layer=Layer.L7_INTENTION.value,
                        user_id=user_id,
                        app_ids=app_ids,
                        embedding=embedding,
                        top_k=math.ceil(intention_limit * _OVERFETCH),
                    )
                    if g.score >= min_score
                ]

        profile_nodes = self._quota_nodes(
            profile_nodes, math.ceil(effective_profile_limit * _OVERFETCH)
        )

        expanded_facts = self._graph_expand_evidence(
            graph_schema_hits, app_ids=app_ids, user_id=user_id
        )
        seen_normal_ids = {n.node_id for n in normal_nodes}
        for fact in expanded_facts:
            if fact.node_id in seen_normal_ids:
                continue
            seen_normal_ids.add(fact.node_id)
            normal_nodes.append(fact)

        seen: set[str] = set()
        merged: list[MemoryNode] = []
        for node in [*profile_nodes, *proactive_nodes, *normal_nodes]:
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            merged.append(node)

        deduped = expand_evolution_chains(vector=vector, hits=merged)

        profile_items: list[dict] = []
        proactive_items: list[dict] = []
        normal_items: list[dict] = []
        for item in deduped:
            lv = item["node"].layer.value
            if lv in _PROFILE_VALS:
                profile_items.append(item)
            elif lv in _PROACTIVE_VALS:
                proactive_items.append(item)
            else:
                normal_items.append(item)

        normal_items = self._rerank_normal(query, normal_items)

        profile_items = _profile_quota_select(
            profile_items, effective_profile_limit, _IDENTITY_VALS, _SCHEMA_VALS
        )
        proactive_items = proactive_items[:intention_limit] if intention_limit > 0 else []
        normal_items = normal_items[:limit]

        logger.info(
            "read_legacy user=%s profile=%d normal=%d proactive=%d",
            user_id, len(profile_items), len(normal_items), len(proactive_items),
        )

        return SearchMemories(
            profile=[self._item_to_memory(it) for it in profile_items],
            proactive=[self._item_to_memory(it) for it in proactive_items],
            normal=[self._item_to_memory(it) for it in normal_items],
        )

    # ---- Internal helpers ----------------------------------------------------------

    def _graph_expand_evidence(
        self,
        schema_hits: list[MemoryNode],
        *,
        app_ids: list[str],
        user_id: str,
    ) -> list[MemoryNode]:
        """Follow DERIVED_FROM edges from each schema hit to surface its evidence fact ids."""
        graph = self.factory.graph
        if graph is None or not schema_hits:
            return []
        seen: set[str] = set()
        out: list[MemoryNode] = []
        for schema in schema_hits[:5]:
            try:
                evidence_ids = graph.evidence_of(schema.node_id)
            except Exception:
                continue
            for fid in evidence_ids:
                if fid in seen:
                    continue
                seen.add(fid)
                node = self.factory.vector.get(fid)
                if node is None or node.status != MemoryStatus.ACTIVE:
                    continue
                if node.user_id != user_id or node.app_id not in app_ids:
                    continue
                # Inherit a fraction of the schema's score so it ranks but doesn't dominate.
                node.score = max(node.score, schema.score * 0.7)
                out.append(node)
        return out

    @staticmethod
    def _graph_to_node(g) -> MemoryNode:
        """Adapt a GraphNode from the graph store into a scored MemoryNode."""
        node = MemoryNode(
            content=g.content,
            layer=Layer(g.layer),
            app_id=g.app_id,
            user_id=g.user_id,
            agent_id=g.agent_id,
            tags=g.tags,
            node_id=g.node_id,
            gmt_created=g.gmt_created,
        )
        node.score = g.score
        return node

    @staticmethod
    def _quota_nodes(nodes: list[MemoryNode], total_limit: int) -> list[MemoryNode]:
        """Apply the profile identity/schema/free quota to a list of nodes."""
        items = [{"node": n, "score": n.score} for n in nodes]
        selected = _profile_quota_select(
            items, total_limit, _IDENTITY_VALS, _SCHEMA_VALS
        )
        return [it["node"] for it in selected]

    @staticmethod
    def _rerank_normal(query: str, items: list[dict]) -> list[dict]:
        """Rerank normal-route items via intent-weighted RRF over vector and BM25 channels."""
        if len(items) <= 1:
            return items
        weights = INTENT_WEIGHTS_2CHANNEL[classify_intent(query)]
        vec_hits = [
            {"node_id": it["node"].node_id, "node": it["node"], "score": it["score"]}
            for it in sorted(items, key=lambda x: x["score"], reverse=True)
        ]
        bm25_hits: list[dict] = []
        keywords = extract_keywords(query)
        if keywords:
            ranked = bm25.score_and_rank(
                keywords, [(it["node"].node_id, it["node"].content) for it in items]
            )
            bm25_hits = [{"node_id": nid, "score": s} for nid, s in ranked if s > 0]
        fused = rrf.rrf_fuse({"vec": vec_hits, "bm25": bm25_hits}, weights=weights)
        by_id = {it["node"].node_id: it for it in items}
        return [by_id[f["node_id"]] for f in fused if f["node_id"] in by_id]

    @staticmethod
    def _item_to_memory(item: dict) -> MemoryItem:
        """Convert an internal scored item (with chain) into a MemoryItem dataclass."""
        node = item["node"]
        chain = item.get("evolution_chain")
        evolution_chain = (
            [
                EvolutionItem(
                    node_id=entry["node_id"],
                    content=entry["content"],
                    layer=entry["layer"],
                    memory_at=entry.get("memory_at"),
                    gmt_created=entry.get("gmt_created"),
                    speculate=entry.get("speculate"),
                )
                for entry in chain
            ]
            if chain
            else None
        )
        return MemoryItem(
            memory_id=node.node_id,
            content=node.content,
            category=node.category.value,
            score=item["score"],
            tags=list(node.tags),
            memory_at=node.memory_at,
            gmt_created=node.gmt_created,
            gmt_modified=node.gmt_modified,
            evolution_chain=evolution_chain,
        )

    @staticmethod
    def memory_node_to_item(node: MemoryNode) -> MemoryItem:
        """Convert a bare MemoryNode (e.g. from list/get) into a MemoryItem dataclass."""
        return MemoryItem(
            memory_id=node.node_id,
            content=node.content,
            category=node.category.value,
            score=node.score,
            tags=list(node.tags),
            memory_at=node.memory_at,
            gmt_created=node.gmt_created,
            gmt_modified=node.gmt_modified,
        )


def _swallow(task: asyncio.Task) -> None:
    """Drop any exception raised in a fire-and-forget reconsolidation task."""
    if task.cancelled():
        return
    task.exception()
