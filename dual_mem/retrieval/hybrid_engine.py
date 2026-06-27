# -*- coding: utf-8 -*-
"""
Hybrid read engine (Hy-Memory parity, latency-optimized).

Semantic recall + in-pool BM25 rerank + graph evidence fusion. The keyword signal is a
re-rank over the already-recalled semantic pool (no full-collection BM25 scan), so an
exact-term / rare-word / numeric match that semantic ranked low gets surfaced without a
second store round-trip. profile = L0 (VDB) + L6 (graph forward/reverse RRF); L4 identity
routes to normal; L7 to proactive when intention_limit > 0.
"""
import asyncio
import logging

from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.bm25 import compute_bm25_scores, tokenize
from dual_mem.retrieval.evolution import expand_evolution_chains
from dual_mem.retrieval.hybrid_scoring import compute_evidence_boost, score_vdb_node
from dual_mem.retrieval.intention_recall import recall_intentions
from dual_mem.retrieval.profile_evidence import reverse_lookup_l6
from dual_mem.retrieval import rrf
from dual_mem.sdk_models import EvolutionItem, MemoryItem, SearchMemories
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.retrieval.hybrid")

# Hybrid layer routing
_PROFILE_LAYERS = [Layer.L0_BASIC_INFO, Layer.L6_SCHEMA]
_VDB_PROFILE_LAYERS = [Layer.L0_BASIC_INFO]
_VDB_RECALL_LAYERS = [Layer.L2_FACT, Layer.L3_SUMMARY, Layer.L4_IDENTITY]
_PROFILE_RECALL_LIMIT = 10
_PROFILE_LAYER_VALS = {layer.value for layer in _PROFILE_LAYERS}
_NORMAL_SOURCES = {"vdb"}


def _layer_of(item: dict) -> str:
    node = item.get("node")
    if node is not None and node.layer is not None:
        return node.layer.value
    return item.get("layer") or ""


def _l6_to_item(graph_hit: dict) -> dict:
    """Wrap a graph-only L6 hit as an item dict with a synthetic MemoryNode."""
    node = MemoryNode(
        content=graph_hit.get("content", ""),
        layer=Layer.L6_SCHEMA,
        app_id="",
        user_id="",
        node_id=graph_hit["node_id"],
    )
    node.score = graph_hit.get("score", 0.0)
    return {**graph_hit, "node": node, "score": node.score}


def _item_to_memory(item: dict) -> MemoryItem:
    if item.get("node") is None:
        item = _l6_to_item(item)
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


async def search_hybrid(
    *,
    factory: ComponentFactory,
    query: str,
    query_embedding: list[float],
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
    suppress_derived: bool = False,
) -> SearchMemories:
    """Run hybrid recall + fusion; return profile/proactive/normal groups."""
    vector = factory.vector
    graph = factory.graph
    settings = factory.settings

    w_sem = settings.hybrid_w_sem
    w_bm25 = settings.hybrid_w_bm25
    ev_boost_max = settings.hybrid_evidence_boost_max
    ev_saturate = settings.hybrid_evidence_saturate
    # min_score=0 means no gate (benchmark default); only enforce a floor when > 0.
    min_score_gate = min_score if min_score > 0 else 0.0

    final_limit = limit if limit > 0 else 10
    vdb_sem_limit = max(final_limit * 3, 30)
    graph_limit = max(final_limit * 2, 20)

    recall_statuses = [MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED]

    normal_where = build_filter(
        app_ids=app_ids,
        user_id=user_id,
        agent_ids=agent_ids,
        session_ids=session_ids,
        layers=_VDB_RECALL_LAYERS,
        statuses=recall_statuses,
        created_after=created_after,
    )
    profile_where = build_filter(
        app_ids=app_ids,
        user_id=user_id,
        agent_ids=agent_ids,
        session_ids=session_ids,
        layers=_VDB_PROFILE_LAYERS,
        statuses=[MemoryStatus.ACTIVE],
        created_after=created_after,
    )

    async def _vdb_semantic() -> list[dict]:
        nodes = await asyncio.to_thread(
            vector.query,
            embedding=query_embedding,
            where=normal_where,
            top_k=vdb_sem_limit,
        )
        return [{"node_id": n.node_id, "score": n.score, "node": n} for n in nodes]

    async def _profile_vdb() -> list[dict]:
        nodes = await asyncio.to_thread(
            vector.query,
            embedding=query_embedding,
            where=profile_where,
            top_k=_PROFILE_RECALL_LIMIT,
        )
        return [
            {"node_id": n.node_id, "score": n.score, "node": n}
            for n in nodes
            if n.score >= profile_min_score
        ]

    async def _graph_schema() -> list[dict]:
        if graph is None:
            return []
        hits = await asyncio.to_thread(
            graph.query_by_embedding,
            layer=Layer.L6_SCHEMA.value,
            user_id=user_id,
            app_ids=app_ids,
            embedding=query_embedding,
            top_k=graph_limit,
        )
        return [
            {
                "node_id": g.node_id,
                "score": g.score,
                "content": g.content,
                "layer": g.layer,
                "confidence": g.score,
            }
            for g in hits
            if g.score >= profile_min_score
        ]

    results = await asyncio.gather(
        _vdb_semantic(),
        _profile_vdb(),
        _graph_schema(),
        return_exceptions=True,
    )

    vdb_semantic_hits = results[0] if not isinstance(results[0], Exception) else []
    profile_hits = results[1] if not isinstance(results[1], Exception) else []
    graph_schema_hits = results[2] if not isinstance(results[2], Exception) else []

    if isinstance(results[0], Exception):
        logger.warning("[hybrid] vdb semantic failed: %s", results[0])

    intention_hits: list[dict] = []
    if intention_limit > 0:
        intention_hits = await recall_intentions(
            vector,
            query_embedding,
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            limit=intention_limit,
        )
        if graph is not None:
            graph_intentions = await asyncio.to_thread(
                graph.query_by_embedding,
                layer=Layer.L7_INTENTION.value,
                user_id=user_id,
                app_ids=app_ids,
                embedding=query_embedding,
                top_k=intention_limit,
            )
            seen = {h["node_id"] for h in intention_hits}
            for g in graph_intentions:
                if g.node_id in seen:
                    continue
                seen.add(g.node_id)
                node = MemoryNode(
                    content=g.content,
                    layer=Layer.L7_INTENTION,
                    app_id=g.app_id,
                    user_id=g.user_id,
                    agent_id=g.agent_id,
                    tags=g.tags,
                    node_id=g.node_id,
                    gmt_created=g.gmt_created,
                )
                node.score = g.score
                intention_hits.append(
                    {
                        "node_id": g.node_id,
                        "score": g.score,
                        "node": node,
                        "layer": Layer.L7_INTENTION.value,
                        "source": "graph_intention",
                    }
                )
            intention_hits.sort(key=lambda x: x["score"], reverse=True)
            intention_hits = intention_hits[:intention_limit]

    # BM25 re-rank over the recalled semantic pool only (no full-collection scan). A node
    # with a strong exact-term match but a weak embedding similarity is rescued here; we
    # fuse first and gate on the fused score afterwards so the keyword signal is not lost.
    terms = tokenize(query)
    bm25_by_nid: dict[str, float] = {}
    if terms and vdb_semantic_hits:
        contents = [
            f"{h['node'].content} {' '.join(h['node'].tags)}".strip()
            for h in vdb_semantic_hits
        ]
        raw = compute_bm25_scores(terms, contents)
        max_raw = max(raw) if raw else 0.0
        if max_raw > 0:
            for h, r in zip(vdb_semantic_hits, raw):
                if r > 0:
                    bm25_by_nid[h["node_id"]] = r / max_raw
    has_bm25 = bool(bm25_by_nid)

    vdb_scored: list[dict] = []
    for row in vdb_semantic_hits:
        nid = row["node_id"]
        sem_score = row["score"]
        bm25_score = bm25_by_nid.get(nid, 0.0)
        final = score_vdb_node(sem_score, bm25_score, w_sem, w_bm25) if has_bm25 else sem_score
        vdb_scored.append(
            {
                "node_id": nid,
                "node": row["node"],
                "score": final,
                "source": "vdb",
                "_semantic": sem_score,
                "_bm25": bm25_score,
            }
        )
    vdb_scored.sort(key=lambda x: x["score"], reverse=True)

    # Forward L6: batch the evidence-count lookup into a single graph round-trip instead of
    # one evidence_of() call per schema (the previous serial loop dominated read latency).
    forward_l6: list[dict] = []
    if graph is not None and graph_schema_hits:
        schema_ids = [row["node_id"] for row in graph_schema_hits]
        try:
            ev_counts = await asyncio.to_thread(graph.evidence_counts, schema_ids)
        except Exception as exc:
            logger.warning("[hybrid] evidence_counts failed, skipping boost: %s", exc)
            ev_counts = {}
        for row in graph_schema_hits:
            ev_count = ev_counts.get(row["node_id"], 0)
            ev_boost = compute_evidence_boost(ev_count, ev_saturate, ev_boost_max)
            internal_score = row["score"] * (1.0 + ev_boost)
            forward_l6.append(
                {
                    "node_id": row["node_id"],
                    "node": None,
                    "content": row.get("content", ""),
                    "score": row["score"],
                    "source": "profile_forward",
                    "layer": row.get("layer", Layer.L6_SCHEMA.value),
                    "confidence": row.get("confidence", row["score"]),
                    "_internal": internal_score,
                    "_evidence_count": ev_count,
                }
            )
        forward_l6.sort(key=lambda x: x["_internal"], reverse=True)

    reverse_l6: list[dict] = []
    if graph is not None and vdb_scored:
        reverse_l6 = await reverse_lookup_l6(
            graph,
            [v["node_id"] for v in vdb_scored],
            limit=graph_limit,
        )

    fused_l6: list[dict] = []
    if forward_l6 or reverse_l6:
        by_id: dict[str, dict] = {}
        for item in forward_l6 + reverse_l6:
            by_id.setdefault(item["node_id"], item)
        fused = rrf.rrf_fuse({"forward": forward_l6, "reverse": reverse_l6})
        for f_item in fused:
            src = by_id.get(f_item["node_id"], {})
            fused_l6.append(
                {
                    "node_id": f_item["node_id"],
                    "node": None,
                    "content": src.get("content", ""),
                    "layer": src.get("layer", Layer.L6_SCHEMA.value),
                    "score": float(f_item.get("rrf_score", 0.0)),
                    "source": "profile_l6",
                    "confidence": src.get("confidence"),
                }
            )

    expandable = vdb_scored + list(profile_hits)
    meta_by_nid: dict[str, dict] = {it["node_id"]: it for it in expandable if it.get("node_id")}
    if expandable:
        for item in expandable:
            if item.get("node") is not None:
                item["node"].score = item.get("score", 0.0)
        expanded_raw = expand_evolution_chains(
            vector=vector,
            hits=[it["node"] for it in expandable if it.get("node")],
        )
        expanded: list[dict] = []
        for exp in expanded_raw:
            nid = exp["node"].node_id
            orig = meta_by_nid.get(nid, {})
            merged = {
                "node": exp["node"],
                "score": max(exp.get("score", 0.0), orig.get("score", 0.0)),
                "source": orig.get("source", "vdb"),
            }
            if exp.get("evolution_chain"):
                merged["evolution_chain"] = exp["evolution_chain"]
                merged["is_evolved"] = True
            expanded.append(merged)
    else:
        expanded = []

    normal_results = [it for it in expanded if _layer_of(it) not in _PROFILE_LAYER_VALS]
    if min_score_gate > 0:
        normal_results = [
            it
            for it in normal_results
            if it.get("source") not in _NORMAL_SOURCES
            or it.get("score", 0.0) >= min_score_gate
        ]
    normal_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    normal_results = normal_results[:final_limit]

    l0_results = [it for it in expanded if _layer_of(it) in _PROFILE_LAYER_VALS]
    l0_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    profile_results = l0_results + ([] if suppress_derived else fused_l6)
    if profile_limit == 0:
        profile_results = []
    elif profile_limit > 0:
        profile_results = profile_results[:profile_limit]

    if suppress_derived:
        intention_hits = []

    logger.info(
        "read_hybrid user=%s vdb_sem=%d bm25_hits=%d profile_l0=%d fwd_l6=%d rev_l6=%d "
        "normal=%d profile=%d intention=%d",
        user_id,
        len(vdb_semantic_hits),
        len(bm25_by_nid),
        len(profile_hits),
        len(forward_l6),
        len(reverse_l6),
        len(normal_results),
        len(profile_results),
        len(intention_hits),
    )

    return SearchMemories(
        profile=[_item_to_memory(it) for it in profile_results],
        proactive=[_item_to_memory(it) for it in intention_hits],
        normal=[_item_to_memory(it) for it in normal_results],
    )
