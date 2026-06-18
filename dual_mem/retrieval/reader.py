import math

from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval import bm25, rrf
from dual_mem.retrieval.evolution import expand_evolution_chains
from dual_mem.retrieval.intent import (
    INTENT_WEIGHTS_2CHANNEL,
    classify_intent,
    extract_keywords,
)
from dual_mem.types import Layer, MemoryNode, MemoryStatus

PROFILE_LAYERS = [Layer.L0_BASIC_INFO, Layer.L4_IDENTITY]
PROACTIVE_LAYERS = [Layer.L7_INTENTION]
NORMAL_LAYERS = [Layer.L2_FACT, Layer.L5_KNOWLEDGE, Layer.L3_SUMMARY, Layer.L1_RAW]

_OVERFETCH = 1.5
_PROFILE_FULL = 100

_IDENTITY_VALS = {Layer.L0_BASIC_INFO.value, Layer.L4_IDENTITY.value}
_SCHEMA_VALS = {Layer.L6_SCHEMA.value}


def _profile_quota_select(
    items: list[dict],
    total_limit: int,
    identity_vals: set[str],
    schema_vals: set[str],
) -> list[dict]:
    """从 profile 结果池按 identity(L0/L4) 40% / schema(L6) 40% / 自由竞争 20% 选取。

    items 含 ``node`` / ``score``；返回按 score 降序。M4 无图，schema 池基本为空，
    几乎全部进 identity 侧。
    """
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
    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    def search(
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
        created_after: int | None = None,
    ) -> dict:
        embedding = self.factory.embed.embed(query)
        vector = self.factory.vector

        effective_profile_limit = profile_limit if profile_limit > 0 else _PROFILE_FULL

        # ── 三路并行向量召回（over-fetch 1.5x，演化展开后再截取）──
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

        # proactive 路 = L7_INTENTION，存在图库中；M4 无图 → 恒空
        proactive_nodes: list[MemoryNode] = []

        # profile 配额（identity/schema/自由）
        profile_nodes = self._quota_nodes(
            profile_nodes, math.ceil(effective_profile_limit * _OVERFETCH)
        )

        # ── 合并三路（profile > proactive > normal），node_id 去重 ──
        seen: set[str] = set()
        merged: list[MemoryNode] = []
        for node in [*profile_nodes, *proactive_nodes, *normal_nodes]:
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            merged.append(node)

        # ── 演化链展开 + 去重 ──
        deduped = expand_evolution_chains(vector=vector, hits=merged)

        # ── 按层分回三组 ──
        profile_set = {layer.value for layer in PROFILE_LAYERS}
        proactive_set = {layer.value for layer in PROACTIVE_LAYERS}
        profile_items: list[dict] = []
        proactive_items: list[dict] = []
        normal_items: list[dict] = []
        for item in deduped:
            lv = item["node"].layer.value
            if lv in profile_set:
                profile_items.append(item)
            elif lv in proactive_set:
                proactive_items.append(item)
            else:
                normal_items.append(item)

        # ── normal 路 rerank（intent 权重 + vec/bm25 双通道 RRF）──
        normal_items = self._rerank_normal(query, normal_items)

        # ── 各路截取 ──
        profile_items = _profile_quota_select(
            profile_items, effective_profile_limit, _IDENTITY_VALS, _SCHEMA_VALS
        )
        proactive_items = []  # 无图，恒空
        normal_items = normal_items[:limit]

        return {
            "profile": [self._item_to_dict(it) for it in profile_items],
            "proactive": [self._item_to_dict(it) for it in proactive_items],
            "normal": [self._item_to_dict(it) for it in normal_items],
        }

    @staticmethod
    def _quota_nodes(nodes: list[MemoryNode], total_limit: int) -> list[MemoryNode]:
        items = [{"node": n, "score": n.score} for n in nodes]
        selected = _profile_quota_select(
            items, total_limit, _IDENTITY_VALS, _SCHEMA_VALS
        )
        return [it["node"] for it in selected]

    @staticmethod
    def _rerank_normal(query: str, items: list[dict]) -> list[dict]:
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
        return [by_id[f["node_id"]] for f in fused]

    @staticmethod
    def _item_to_dict(item: dict) -> dict:
        return Reader._to_dict(
            item["node"],
            score=item["score"],
            evolution_chain=item["evolution_chain"],
        )

    @staticmethod
    def _to_dict(
        node: MemoryNode,
        score: float | None = None,
        evolution_chain: list | None = None,
    ) -> dict:
        result = {
            "memory_id": node.node_id,
            "content": node.content,
            "category": node.category.value,
            "score": round(node.score if score is None else score, 4),
            "tags": node.tags,
            "memory_at": node.memory_at,
            "gmt_created": node.gmt_created,
            "gmt_modified": node.gmt_modified,
        }
        if evolution_chain:
            result["evolution_chain"] = evolution_chain
        return result
