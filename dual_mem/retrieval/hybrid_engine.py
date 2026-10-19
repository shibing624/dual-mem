# -*- coding: utf-8 -*-
"""
Hybrid read engine (Hy-Memory parity, latency-optimized).

Semantic and lexical recall + calibrated BM25 rerank + graph evidence fusion.
Lexical recall admits exact-term matches outside the semantic candidate pool;
both routes use the same semantic/BM25 scoring scale. profile = L2 (VDB) + L3
(graph forward/reverse RRF); L2 identity
routes to normal; intentions to proactive when intention_limit > 0.
"""
import asyncio
import logging
import math
import re
from dataclasses import replace

from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.bm25 import (
    compute_bm25_confidences,
    compute_bm25_scores,
    tokenize,
)
from dual_mem.retrieval.evolution import expand_evolution_chains
from dual_mem.retrieval.hybrid_scoring import compute_evidence_boost, score_vdb_node
from dual_mem.retrieval.intention_recall import recall_intentions
from dual_mem.retrieval.profile_evidence import reverse_lookup_derived
from dual_mem.retrieval import rrf
from dual_mem.sdk_models import EvolutionItem, MemoryItem, SearchMemories
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.retrieval.hybrid")

# Hybrid layer routing
_PROFILE_LAYERS = [Layer.L2_PROFILE, Layer.L3_DERIVED]
_VDB_PROFILE_LAYERS = [Layer.L2_PROFILE, Layer.L3_DERIVED]
_VDB_RECALL_LAYERS = [Layer.L1_FACT, Layer.L2_PROFILE]
# Note: L0_RAW excluded from VDB semantic recall (align hy_memory: raw fragments
#   are noisy and dilute top-k ranking; only extracted/curated layers are recalled).
_PROFILE_RECALL_LIMIT = 10
_PROFILE_LAYER_VALS = {layer.value for layer in _PROFILE_LAYERS}
_NORMAL_SOURCES = {"vdb"}
# RRF over the raw route only ranks raw hits among themselves; this scale keeps
# raw scores below curated memories of equal rank when routes are flattened.
_RAW_SCORE_SCALE = 0.5


def _layer_of(item: dict) -> str:
    node = item.get("node")
    if node is not None and node.layer is not None:
        return node.layer.value
    return item.get("layer") or ""


def _derived_to_item(graph_hit: dict) -> dict:
    """Wrap a derived graph hit as an item dict with a synthetic MemoryNode."""
    node = MemoryNode(
        content=graph_hit.get("content", ""),
        layer=Layer.L3_DERIVED,
        app_id="",
        user_id="",
        node_id=graph_hit["node_id"],
    )
    node.score = graph_hit.get("score", 0.0)
    return {**graph_hit, "node": node, "score": node.score}


def _valid_at(node: MemoryNode | None, as_of: int | None) -> bool:
    """Whether an explicit validity window contains ``as_of`` (no window = valid)."""
    if node is None or as_of is None:
        return True
    if node.valid_until is not None and node.valid_until < as_of:
        return False
    if node.valid_from is not None and node.valid_from > as_of:
        return False
    return True


def _brief(node: MemoryNode) -> EvolutionItem:
    return EvolutionItem(
        node_id=node.node_id,
        content=node.content,
        layer=node.layer.value,
        memory_at=node.memory_at,
        gmt_created=node.gmt_created,
    )


def attach_relations(vector, items: list[MemoryItem]) -> None:
    """Fill one-hop ``previous`` (direct supersedes) and current ``conflicts``."""
    wanted = {
        nid for item in items for nid in [*item.supersedes, *item.conflicts_with]
    }
    if not wanted:
        return
    found = vector.get_by_ids(list(wanted))
    for item in items:
        previous = [found[n] for n in item.supersedes if n in found]
        if previous:
            item.previous = [_brief(n) for n in previous]
        conflicts = [
            found[n] for n in item.conflicts_with if n in found and found[n].is_latest
        ]
        if conflicts:
            item.conflicts = [_brief(n) for n in conflicts]


def _lexical_candidates(
    *,
    vector,
    where: dict,
    terms: list[str],
    excluded: set[str],
    limit: int,
    as_of: int | None,
) -> list[MemoryNode]:
    """Best BM25 matches over every current memory of the scope not already recalled.

    Memory scopes are per user (a few hundred to a few thousand nodes), so scoring them
    in one pass is cheap and needs no index. Only nodes sharing a query term are
    returned, best first.
    """
    nodes = [n for n in vector.get_many(where, limit=20_000) if _valid_at(n, as_of)]
    if not nodes:
        return []
    scores = compute_bm25_scores(
        terms, [f"{n.content} {' '.join(n.tags)}".strip() for n in nodes]
    )
    ranked = sorted(zip(nodes, scores), key=lambda pair: pair[1], reverse=True)
    return [
        node for node, score in ranked
        if score > 0 and node.node_id not in excluded
    ][:limit]


def _cosine_score(query: list[float], embedding: list[float] | None) -> float:
    if not embedding or len(query) != len(embedding):
        return 0.0
    norm = math.sqrt(sum(x * x for x in query) * sum(x * x for x in embedding))
    return sum(a * b for a, b in zip(query, embedding)) / norm if norm else 0.0


def _source_excerpt(content: str, query: str, anchors: list[str], max_chars: int = 3200) -> str:
    """Select verbatim, role-labelled passages instead of head-truncating a source."""
    if len(content) <= max_chars:
        return content
    turns = re.split(r"(?m)(?=^\[(?:user|assistant|tool|system)\]:)", content)
    passages: list[str] = []
    roles: list[str] = []
    for turn in turns:
        if not turn.strip():
            continue
        label = re.match(r"\[(\w+)\]:", turn)
        role = label.group(1) if label else ""
        start = 0
        while start < len(turn):
            end = min(start + 900, len(turn))
            if end < len(turn):
                boundary = turn.rfind(" ", start + 450, end)
                if boundary > start:
                    end = boundary
            passage = turn[start:end].strip()
            if start and role:
                passage = f"[{role}]: […] {passage}"
            passages.append(passage)
            roles.append(role)
            start = end
    if not passages:
        return content[:max_chars]
    query_scores = compute_bm25_scores(tokenize(query), passages)
    anchor_scores = compute_bm25_scores(tokenize(" ".join(anchors[:4])), passages)
    query_max = max(query_scores, default=0) or 1
    anchor_max = max(anchor_scores, default=0) or 1
    ranked = sorted(range(len(passages)), key=lambda i: -(
        query_scores[i] / query_max + anchor_scores[i] / anchor_max
    ))
    selected: set[int] = set()
    size = 0
    for index in ranked:
        # A quoted answer needs the preceding user turn to resolve references.
        neighbours = [index]
        if roles[index] == "assistant" and index > 0 and roles[index - 1] == "user":
            neighbours.append(index - 1)
        for i in neighbours:
            if i not in selected and size + len(passages[i]) + 8 <= max_chars:
                selected.add(i)
                size += len(passages[i]) + 8
        if len(selected) >= 3:
            break
    return "\n[…]\n".join(passages[i] for i in sorted(selected))


def _item_to_memory(item: dict) -> MemoryItem:
    if item.get("node") is None:
        item = _derived_to_item(item)
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
    raw_ts = (node.custom or {}).get("merged_timestamps")
    merged_timestamps = (
        [int(x) for x in raw_ts if isinstance(x, (int, float))]
        if isinstance(raw_ts, list) and raw_ts
        else None
    )
    update_type = (node.custom or {}).get("update_type") or None
    if update_type is not None:
        update_type = str(update_type)
    source_node_id = (node.custom or {}).get("source_node_id") or None
    if source_node_id is not None:
        source_node_id = str(source_node_id)
    return MemoryItem(
        memory_id=node.node_id,
        layer=node.layer.value,
        kind=node.kind or node.category.value,
        owner=node.owner,
        is_latest=node.is_latest,
        revision=node.revision,
        supersedes=list(node.supersedes),
        source_ids=list((node.custom or {}).get("source_ids", [])),
        conflicts_with=list((node.custom or {}).get("conflicts_with", [])),
        valid_from=node.valid_from,
        valid_until=node.valid_until,
        content=node.content,
        category=node.category.value,
        score=item["score"],
        tags=list(node.tags),
        session_id=node.session_id,
        memory_at=node.memory_at,
        gmt_created=node.gmt_created,
        gmt_modified=node.gmt_modified,
        evolution_chain=evolution_chain,
        merged_timestamps=merged_timestamps,
        update_type=update_type,
        source_node_id=source_node_id,
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
    include_derived: bool = True,
    include_derived_fusion: bool = True,
    as_of: int | None = None,
    raw_limit: int = 0,
    include_relations: bool = True,
) -> SearchMemories:
    """Run hybrid recall + fusion; return profile/proactive/normal/raw groups.

    ``as_of`` drops memories whose explicit ``valid_from``/``valid_until`` window
    excludes that time (a past TEMPORAL exception). ``raw_limit`` adds an L0_RAW
    route as fallback evidence for details extraction did not keep.
    """
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
    candidate_multiplier = max(1.0, float(settings.normal_candidate_multiplier))
    # v10: enlarge semantic recall to ~hy's combined sem+kw pool (45+60=105),
    # compensating for ChromaDB's lack of native keyword index. L0_RAW excluded.
    vdb_sem_limit = max(int(math.ceil(final_limit * candidate_multiplier * 2.5)), 60)
    # Extra candidates admitted from a lexical pass over the scope (pure keyword recall).
    lexical_limit = max(final_limit // 2, 10)
    graph_limit = max(final_limit * 2, 20)

    use_derived = include_derived
    # Current facts/profiles serve normal recall; derived evidence uses profile fusion.
    recall_statuses = [MemoryStatus.ACTIVE]

    normal_where = build_filter(
        app_ids=app_ids,
        user_id=user_id,
        agent_ids=agent_ids,
        session_ids=session_ids,
        layers=_VDB_RECALL_LAYERS + ([Layer.L3_DERIVED] if use_derived else []),
        statuses=recall_statuses,
        latest_only=True,
        created_after=created_after,
    )
    profile_where = build_filter(
        app_ids=app_ids,
        user_id=user_id,
        agent_ids=agent_ids,
        session_ids=session_ids,
        layers=_VDB_PROFILE_LAYERS if include_derived else [Layer.L2_PROFILE],
        statuses=[MemoryStatus.ACTIVE],
        latest_only=True,
        created_after=created_after,
    )

    async def _vdb_semantic() -> list[dict]:
        nodes = await asyncio.to_thread(
            vector.query,
            embedding=query_embedding,
            where=normal_where,
            top_k=vdb_sem_limit,
        )
        return [
            {"node_id": n.node_id, "score": n.score, "node": n}
            for n in nodes
            if _valid_at(n, as_of)
        ]

    async def _profile_vdb() -> list[dict]:
        if profile_limit == 0:
            return []
        nodes = await asyncio.to_thread(
            vector.query,
            embedding=query_embedding,
            where=profile_where,
            top_k=_PROFILE_RECALL_LIMIT,
        )
        return [
            {"node_id": n.node_id, "score": n.score, "node": n}
            for n in nodes
            if n.score >= profile_min_score and _valid_at(n, as_of)
        ]

    async def _raw_route() -> list[dict]:
        if raw_limit <= 0:
            return []
        where = build_filter(
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            layers=[Layer.L0_RAW],
            statuses=[MemoryStatus.ACTIVE],
            created_after=created_after,
        )
        nodes = await asyncio.to_thread(
            vector.query, embedding=query_embedding, where=where, top_k=raw_limit * 3
        )
        lexical_nodes = await asyncio.to_thread(
            _lexical_candidates, vector=vector, where=where, terms=tokenize(query),
            excluded={n.node_id for n in nodes}, limit=raw_limit * 2, as_of=as_of,
        )
        if lexical_nodes:
            embedded = await asyncio.to_thread(vector.get_by_ids, [n.node_id for n in lexical_nodes])
            for candidate in lexical_nodes:
                node = embedded.get(candidate.node_id, candidate)
                node.score = _cosine_score(query_embedding, node.embedding)
                nodes.append(node)
        if not nodes:
            return []
        semantic = sorted(nodes, key=lambda n: n.score, reverse=True)
        lexical = sorted(
            zip(nodes, compute_bm25_scores(tokenize(query), [n.content for n in nodes])),
            key=lambda pair: pair[1],
            reverse=True,
        )
        fused = rrf.rrf_fuse(
            {
                "semantic": [{"node_id": n.node_id} for n in semantic],
                "bm25": [{"node_id": n.node_id} for n, s in lexical if s > 0],
            }
        )
        by_id = {n.node_id: n for n in nodes}
        top = max((f["rrf_score"] for f in fused), default=0.0) or 1.0
        return [
            {
                "node_id": f["node_id"],
                "node": by_id[f["node_id"]],
                "score": _RAW_SCORE_SCALE * float(f["rrf_score"]) / top,
                "source": "raw",
            }
            for f in fused[:raw_limit]
        ]

    async def _graph_schema() -> list[dict]:
        if graph is None or not include_derived:
            return []
        hits = await asyncio.to_thread(
            graph.query_by_embedding,
            layer=Layer.L3_DERIVED.value,
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
                "node": head,
            }
            for g in hits
            if g.score >= profile_min_score
            and (head := vector.get(g.node_id)) is not None
            and head.is_latest
            and head.app_id in app_ids
            and head.user_id == user_id
            and head.agent_id in (agent_ids or [head.agent_id])
            and (not session_ids or head.session_id in session_ids)
            and (created_after is None or head.gmt_created >= created_after)
            and _valid_at(head, as_of)
        ]

    results = await asyncio.gather(
        _vdb_semantic(),
        _profile_vdb(),
        _graph_schema(),
        _raw_route(),
        return_exceptions=True,
    )

    vdb_semantic_hits = results[0] if not isinstance(results[0], Exception) else []
    profile_hits = results[1] if not isinstance(results[1], Exception) else []
    graph_schema_hits = results[2] if not isinstance(results[2], Exception) else []
    raw_hits: list[dict] = results[3] if isinstance(results[3], list) else []

    if isinstance(results[0], Exception):
        logger.warning("[hybrid] vdb semantic failed: %s", results[0])
    if isinstance(results[1], Exception):
        logger.warning("[hybrid] profile vdb failed: %s", results[1])
    if isinstance(results[2], Exception):
        logger.warning("[hybrid] graph schema failed: %s", results[2])
    if isinstance(results[3], Exception):
        logger.warning("[hybrid] raw route failed: %s", results[3])

    intention_hits: list[dict] = []
    if use_derived and intention_limit > 0:
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
                layer=Layer.L3_DERIVED.value,
                user_id=user_id,
                app_ids=app_ids,
                embedding=query_embedding,
                top_k=intention_limit,
            )
            seen = {h["node_id"] for h in intention_hits}
            for g in graph_intentions:
                head = vector.get(g.node_id)
                if head is None or not head.is_latest or head.kind != "intention":
                    continue
                if agent_ids and head.agent_id not in agent_ids:
                    continue
                if g.node_id in seen:
                    continue
                seen.add(g.node_id)
                node = MemoryNode(
                    content=g.content,
                    layer=head.layer,
                    kind=head.kind,
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
                        "layer": head.layer.value,
                        "source": "graph_intention",
                    }
                )
            intention_hits.sort(key=lambda x: x["score"], reverse=True)
            intention_hits = intention_hits[:intention_limit]
        intention_hits = [h for h in intention_hits if _valid_at(h["node"], as_of)]

    # Every route uses the same candidate calibration. A second profile lookup must
    # not overwrite a fused fact/profile score with its larger raw cosine score.
    seen_candidates = {h["node_id"] for h in vdb_semantic_hits}
    for hit in profile_hits + graph_schema_hits:
        if hit["node_id"] not in seen_candidates:
            vdb_semantic_hits.append(hit)
            seen_candidates.add(hit["node_id"])

    # Admit lexical matches before calibration so both routes have identical BM25
    # statistics. Raw BM25 scores cannot be fused with cosine similarities.
    terms = tokenize(query)
    bm25_by_nid: dict[str, float] = {}
    if terms and w_bm25 > 0:
        lexical = await asyncio.to_thread(
            _lexical_candidates,
            vector=vector,
            where=normal_where,
            terms=terms,
            excluded={h["node_id"] for h in vdb_semantic_hits},
            limit=lexical_limit,
            as_of=as_of,
        )
        lexical_nodes = await asyncio.to_thread(
            vector.get_by_ids, [n.node_id for n in lexical]
        ) if lexical else {}
        for candidate in lexical:
            node = lexical_nodes.get(candidate.node_id, candidate)
            vdb_semantic_hits.append(
                {
                    "node_id": node.node_id,
                    "score": _cosine_score(query_embedding, node.embedding),
                    "node": node,
                }
            )
        contents = [
            f"{h['node'].content} {' '.join(h['node'].tags)}".strip()
            for h in vdb_semantic_hits
        ]
        calibrated = compute_bm25_confidences(
            terms, contents, compute_bm25_scores(terms, contents)
        )
        bm25_by_nid = {
            hit["node_id"]: score
            for hit, score in zip(vdb_semantic_hits, calibrated, strict=True)
            if score > 0
        }
    has_bm25 = bool(bm25_by_nid)

    vdb_scored: list[dict] = []
    for row in vdb_semantic_hits:
        nid = row["node_id"]
        sem_score = row["score"]
        bm25_score = bm25_by_nid.get(nid, 0.0)
        final = (
            score_vdb_node(sem_score, bm25_score, w_sem, w_bm25)
            if has_bm25
            else sem_score
        )
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
    scored_by_id = {row["node_id"]: row for row in vdb_scored}

    if raw_limit > 0:
        # Extracted memories are also retrieval keys for their source conversations.
        # This recovers surrounding details without a second extraction/model call.
        anchors: dict[str, list[str]] = {}
        source_scores: dict[str, float] = {}
        for row in vdb_scored[:max(final_limit, raw_limit * 4)]:
            node = row["node"]
            source_ids = ((node.custom or {}).get("source_ids") or []) if node.layer == Layer.L3_DERIVED else []
            if not source_ids:
                source_ids = [(node.custom or {}).get("source_node_id")]
            for source in source_ids:
                if not source:
                    continue
                anchors.setdefault(source, []).append(node.content)
                source_scores[source] = max(source_scores.get(source, 0), row["score"])
        sources = await asyncio.to_thread(vector.get_by_ids, list(anchors))
        # L3 sources point to facts; fact sources point to L0. Follow that one hop,
        # while checking scope before reading any linked text into the response.
        for source_id, node in list(sources.items()):
            if (node.app_id not in app_ids or node.user_id != user_id
                    or (agent_ids and node.agent_id not in agent_ids)):
                continue
            raw_id = (node.custom or {}).get("source_node_id") if node.layer != Layer.L0_RAW else None
            if raw_id:
                anchors.setdefault(raw_id, []).extend(anchors[source_id])
                source_scores[raw_id] = max(source_scores.get(raw_id, 0), source_scores[source_id])
        extra_ids = [nid for nid in anchors if nid not in sources]
        if extra_ids:
            sources.update(await asyncio.to_thread(vector.get_by_ids, extra_ids))
        sources = {
            nid: node for nid, node in sources.items()
            if node.layer == Layer.L0_RAW and node.app_id in app_ids and node.user_id == user_id
            and (not agent_ids or node.agent_id in agent_ids)
            and (not session_ids or node.session_id in session_ids)
            and (created_after is None or node.gmt_created >= created_after)
            and _valid_at(node, as_of)
        }
        linked = sorted(sources, key=lambda nid: -source_scores[nid])
        merged = rrf.rrf_fuse({
            "direct": [{"node_id": row["node_id"]} for row in raw_hits],
            "linked": [{"node_id": nid} for nid in linked],
        })
        raw_nodes = {row["node_id"]: row["node"] for row in raw_hits} | sources
        top = max((row["rrf_score"] for row in merged), default=0) or 1
        raw_hits = [
            {"node_id": row["node_id"],
             "node": replace(raw_nodes[row["node_id"]], content=_source_excerpt(
                 raw_nodes[row["node_id"]].content, query, anchors.get(row["node_id"], [])
             )),
             "score": _RAW_SCORE_SCALE * row["rrf_score"] / top, "source": "raw"}
            for row in merged[:raw_limit]
        ]

    # Forward derived: batch the evidence-count lookup into a single graph round-trip instead of
    # one evidence_of() call per schema (the previous serial loop dominated read latency).
    forward_derived: list[dict] = []
    if (
        use_derived
        and include_derived_fusion
        and graph is not None
        and graph_schema_hits
    ):
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
            forward_derived.append(
                {
                    "node_id": row["node_id"],
                    "node": None,
                    "content": row.get("content", ""),
                    "score": row["score"],
                    "source": "profile_forward",
                    "layer": row.get("layer", Layer.L3_DERIVED.value),
                    "confidence": row.get("confidence", row["score"]),
                    "_internal": internal_score,
                    "_evidence_count": ev_count,
                }
            )
        forward_derived.sort(key=lambda x: x["_internal"], reverse=True)

    reverse_derived: list[dict] = []
    if use_derived and include_derived_fusion and graph is not None and vdb_scored:
        reverse_derived = await reverse_lookup_derived(
            graph,
            [v["node_id"] for v in vdb_scored],
            limit=graph_limit,
        )

    fused_derived: list[dict] = []
    if forward_derived or reverse_derived:
        by_id: dict[str, dict] = {}
        for item in forward_derived + reverse_derived:
            head = vector.get(item["node_id"])
            if (
                head is None
                or not head.is_latest
                or head.app_id not in app_ids
                or head.user_id != user_id
                or (agent_ids and head.agent_id not in agent_ids)
                or not _valid_at(head, as_of)
            ):
                continue
            item["node"] = head
            by_id.setdefault(item["node_id"], item)
        fused = rrf.rrf_fuse(
            {
                "forward": [i for i in forward_derived if i["node_id"] in by_id],
                "reverse": [i for i in reverse_derived if i["node_id"] in by_id],
            }
        )
        for f_item in fused:
            src = by_id.get(f_item["node_id"], {})
            scored = scored_by_id.get(f_item["node_id"])
            # RRF orders graph routes internally; its ~0.01 values cannot compete
            # with cosine/BM25 scores when the public routes are flattened.
            relevance = scored["score"] if scored else (
                _cosine_score(query_embedding, src["node"].embedding) * w_sem
                if has_bm25 else _cosine_score(query_embedding, src["node"].embedding)
            )
            fused_derived.append(
                {
                    "node_id": f_item["node_id"],
                    "node": src.get("node"),
                    "content": src.get("content", ""),
                    "layer": src.get("layer", Layer.L3_DERIVED.value),
                    "score": relevance,
                    "source": "profile_derived",
                    "confidence": src.get("confidence"),
                }
            )

    reranker = factory.reranker
    if reranker is not None:
        pool = {row["node_id"]: row for row in vdb_scored + fused_derived}
        curated = sorted(pool.values(), key=lambda row: -row["score"])[
            :max(final_limit, settings.rerank_candidate_limit)
        ]
        candidates = curated + raw_hits
        scores = await reranker.rerank(
            query=query, documents=[row["node"].content for row in candidates]
        )
        score_by_id = {row["node_id"]: score for row, score in zip(candidates, scores, strict=True)}
        # Only cross-encoder scores compete after reranking, never a mix with
        # uncalibrated cosine scores from candidates outside the rerank budget.
        vdb_scored = [row for row in vdb_scored if row["node_id"] in score_by_id]
        fused_derived = [row for row in fused_derived if row["node_id"] in score_by_id]
        for row in vdb_scored + fused_derived + raw_hits:
            row["score"] = score_by_id[row["node_id"]]

    expandable = vdb_scored
    meta_by_nid: dict[str, dict] = {
        it["node_id"]: it for it in expandable if it.get("node_id")
    }
    # NOTE(dual_vs_hy): evolution-chain 展开只作用于 profile/L3 桶 —— normal 路保持
    # ACTIVE 纯事实（S1 原始记忆保真），S2 深层证据（chain/schema）单独进 profile 桶。
    # 实测 62-74% 检索结果被 chain 挤占导致事实丢失，分段后事实路与 s1 一致。
    if use_derived and expandable:
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
            else:
                merged["is_evolved"] = False
            expanded.append(merged)
    else:
        expanded = expandable

    normal_results = [
        it
        for it in expanded
        if _layer_of(it) not in _PROFILE_LAYER_VALS
        or (profile_limit == 0 and _layer_of(it) == Layer.L2_PROFILE.value)
    ]
    if min_score_gate > 0:
        normal_results = [
            it
            for it in normal_results
            if it.get("source") not in _NORMAL_SOURCES
            or it.get("score", 0.0) >= min_score_gate
        ]
    normal_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    normal_results = normal_results[:final_limit]

    profile_results = [it for it in expanded if _layer_of(it) in _PROFILE_LAYER_VALS]
    profile_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    profile_results = list(
        {
            it["node"].node_id: it
            for it in profile_results + (fused_derived if use_derived else [])
            if it.get("node")
        }.values()
    )
    profile_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    if profile_limit == 0:
        profile_results = []
    elif profile_limit > 0:
        profile_results = profile_results[:profile_limit]

    logger.info(
        "read_hybrid user=%s vdb_sem=%d bm25_hits=%d profile_l0=%d fwd_l6=%d rev_l6=%d "
        "normal=%d profile=%d intention=%d",
        user_id,
        len(vdb_semantic_hits),
        len(bm25_by_nid),
        len(profile_hits),
        len(forward_derived),
        len(reverse_derived),
        len(normal_results),
        len(profile_results),
        len(intention_hits),
    )

    memories = SearchMemories(
        profile=[_item_to_memory(it) for it in profile_results],
        proactive=[_item_to_memory(it) for it in intention_hits],
        normal=[_item_to_memory(it) for it in normal_results],
        raw=[_item_to_memory(it) for it in raw_hits],
    )
    if include_relations:
        attach_relations(
            vector, [*memories.profile, *memories.proactive, *memories.normal]
        )
    return memories
