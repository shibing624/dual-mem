# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: System2 cognitive-processing agent (hy-memory ultra style). Phase 1 prepares
clustered fact materials (no LLM); phase 2 hands them to the LLM in ONE single-shot
chat_json call that emits JSON ops (create_schema / add_evidence / add_edge). The ReAct
8-tool multi-turn loop was removed: on a 4B model its search_graph calls all fail
(embed_sync unavailable) and it degrades to search_vdb with random facts, building
schemas from haystack noise (see benchmark badcase 7405e8b1).
"""
import json
import logging
import os

from dual_mem.agent import prompts
from dual_mem.isolation import build_filter
from dual_mem.providers.llm import is_chinese
from dual_mem.registry import ComponentFactory
from dual_mem.system2.clustering import cluster_facts
from dual_mem.system2.s2_tools import System2ToolExecutor
from dual_mem.types import Layer, MemoryStatus

logger = logging.getLogger("dual_mem.system2.agent")


_S2_LAYERS = [Layer.L2_FACT, Layer.L4_IDENTITY]
_GRAPH_TOPK = 8


async def prepare_materials(
    *, factory: ComponentFactory, app_id: str, user_id: str, agent_id: str
) -> dict:
    """Cluster fresh facts and gather existing schemas/tags as System2 input materials."""
    where = build_filter(
        app_ids=[app_id],
        user_id=user_id,
        agent_ids=[agent_id],
        layers=_S2_LAYERS,
        statuses=[MemoryStatus.ACTIVE],
    )
    all_facts = factory.vector.get_many(where)

    fresh_ids = [fact.node_id for fact in all_facts if fact.s2_evidence_count == 0]
    fresh = []
    if fresh_ids:
        # Single batched fetch (with embeddings) instead of one get() per fact.
        full_by_id = factory.vector.get_by_ids(fresh_ids)
        for nid in fresh_ids:
            full = full_by_id.get(nid)
            if full is not None and full.embedding:
                fresh.append(full)

    clusters = cluster_facts(
        fresh,
        stage1_sim=factory.settings.cluster_stage1_sim,
        stage2_sim=factory.settings.cluster_stage2_sim,
    )

    graph = factory.graph
    graph_forward: list[dict] = []
    existing_tags: list[str] = []
    seen: set[str] = set()
    if graph is not None:
        tag_set: set[str] = set()
        for cluster in clusters:
            schemas = graph.query_by_embedding(
                layer=Layer.L6_SCHEMA.value,
                user_id=user_id,
                app_ids=[app_id],
                embedding=cluster["centroid_embedding"],
                top_k=_GRAPH_TOPK,
            )
            for node in schemas:
                if node.node_id in seen:
                    continue
                seen.add(node.node_id)
                graph_forward.append(
                    {"node_id": node.node_id, "content": node.content, "tags": node.tags}
                )
                tag_set.update(node.tags)
        existing_tags = sorted(tag_set)

    # NOTE(dual_vs_hy): hy ultra 建图增强 —— env 开关默认关，不改变现有行为。
    # 1) graph_reverse：用聚类事实 id 反向查引用它的已有 L6 schema，帮 LLM 复用/合并已有 schema
    #    （hy: graph_store.find_referencing_memories(vdb_ids, limit=50)）。
    # 2) unprocessed_facts：聚类之外（noise）的 fresh 事实加菜进建图材料，给 LLM 更多上下文
    #    （hy: _S2_MAX_UNCLUSTERED_FACTS=15）。
    graph_reverse: list[dict] = []
    unprocessed_facts: list[dict] = []
    if clusters:
        if int(os.getenv("DUAL_MEM_EXP_GRAPH_REVERSE", "0")):
            try:
                rev_ids = [
                    f["node_id"] for c in clusters for f in c["facts"]
                ][:100]
                graph_reverse = graph.find_referencing_memories(rev_ids, limit=50) if graph is not None else []
            except Exception as exc:  # noqa: BLE001
                logger.warning("s2 graph_reverse failed: %s", exc)
        topk = int(os.getenv("DUAL_MEM_EXP_UNPROCESSED_TOPK", "0"))
        if topk > 0:
            clustered_ids = {f["node_id"] for c in clusters for f in c["facts"]}
            unprocessed = [f for f in fresh if f.node_id not in clustered_ids]
            unprocessed_facts = [
                {"node_id": f.node_id, "content": f.content, "layer": f.layer.value}
                for f in unprocessed[:topk]
            ]

    return {
        "clusters": clusters,
        "graph_forward": graph_forward,
        "graph_reverse": graph_reverse,
        "unprocessed_facts": unprocessed_facts,
        "existing_tags": existing_tags,
        "stats": {
            "total_facts": len(all_facts),
            "fresh_facts": len(fresh),
            "clusters_found": len(clusters),
            "existing_schemas": len(graph_forward),
        },
    }


def _build_user_message(materials: dict, *, max_facts: int = 0) -> str:
    """Render prepared materials (clusters, existing schemas, tags) into the user prompt.

    ``max_facts`` caps the total number of cluster facts written into the prompt across all
    clusters (0 = unlimited). Clustering can collapse hundreds of facts into a single cluster;
    rendering them all overflows the model context window and the whole digest fails. The
    budget is split evenly across clusters; omitted counts are reported so the model knows
    the sample is partial.
    """
    clusters = materials["clusters"]
    if max_facts and clusters:
        per_cluster = max(1, max_facts // len(clusters))
    else:
        per_cluster = 0
    parts = ["## 聚类结果 / Cluster results"]
    for i, cluster in enumerate(clusters):
        parts.append(f"### Cluster {i}（主题: {cluster['centroid_text']}）")
        facts = cluster["facts"]
        shown = facts[:per_cluster] if per_cluster else facts
        for fact in shown:
            parts.append(f"  - [{fact['layer']}] {fact['content']}  (id={fact['node_id']})")
        omitted = len(facts) - len(shown)
        if omitted > 0:
            parts.append(
                f"  - …（另有 {omitted} 条同主题事实已省略 / {omitted} more similar facts omitted）"
            )

    forward = materials["graph_forward"]
    if forward:
        parts.append("\n## 已有 Schema / Existing schemas")
        for node in forward:
            parts.append(f"  - {node['content']}  (id={node['node_id']})")

    reverse = materials.get("graph_reverse") or []
    if reverse:
        parts.append("\n## 引用这些事实的已有 Schema / Existing schemas referencing these facts")
        for node in reverse:
            parts.append(f"  - {node['content']}  (id={node['node_id']})")

    unproc = materials.get("unprocessed_facts") or []
    if unproc:
        parts.append("\n## 未聚类补充事实 / Unclustered extra facts")
        for fact in unproc:
            parts.append(f"  - [{fact['layer']}] {fact['content']}  (id={fact['node_id']})")

    tags = materials["existing_tags"]
    if tags:
        parts.append("\n## 已有标签 / Existing tags")
        parts.append("  " + ", ".join(tags))

    parts.append(
        "\n请使用提供的工具完成认知加工：先 search_graph 检查已有 schema、必要时调"
        " create_schema/create_intention/add_evidence/add_edge。完成后直接停止调用工具。"
        " / Use the tools to do the cognitive work: search existing schemas first,"
        " then create new ones / link evidence as needed. Stop calling tools when finished."
    )
    return "\n".join(parts)


class System2Agent:
    """ReAct-style cognitive agent: drives an OpenAI function-calling loop over the 8 tools."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    async def run(
        self, *, app_id: str, user_id: str, agent_id: str = ""
    ) -> dict:
        """Process one app/user: prepare materials, run ReAct tool loop, return stats."""
        materials = await prepare_materials(
            factory=self.factory, app_id=app_id, user_id=user_id, agent_id=agent_id
        )
        if not materials["clusters"]:
            logger.info(
                "s2_react user=%s no clusters (fresh=%d total=%d), skip",
                user_id, materials["stats"]["fresh_facts"], materials["stats"]["total_facts"],
            )
            return {
                "created_schemas": 0,
                "created_intentions": 0,
                "evidence_added": 0,
                "edges_added": 0,
            }

        logger.info(
            "s2_react user=%s clusters=%d existing_schemas=%d fresh=%d",
            user_id,
            materials["stats"]["clusters_found"],
            materials["stats"]["existing_schemas"],
            materials["stats"]["fresh_facts"],
        )

        cluster_text = " ".join(
            f["content"]
            for cluster in materials["clusters"]
            for f in cluster["facts"]
        )
        system = prompts.SYSTEM2_OPS_ZH if is_chinese(cluster_text) else prompts.SYSTEM2_OPS_EN
        user = _build_user_message(
            materials, max_facts=self.factory.settings.system2_max_prompt_facts
        )

        executor = System2ToolExecutor(
            factory=self.factory, app_id=app_id, user_id=user_id, agent_id=agent_id
        )
        # NOTE(dual_vs_hy): 对标 hy-memory ultra —— 只用 single-shot JSON ops（一次 chat_json
        # 输出 create_schema/add_evidence/add_edge ops）。删除 ReAct 8-tool 多轮循环：
        # ReAct 在 4B 模型上先 search_graph（embed_sync 不可用全部失败）再退化为
        # search_vdb 随机取记忆建 schema，实测 7405e8b1 建出的 4 个 schema 全是干扰 session
        # 内容（living room/microwave/standing waves），且多轮往返放大错误、成本更高。
        # 大 cluster 用 single-shot 时由 prepare_materials 的 per_cluster 截断保护。
        await self._run_single_shot(system=system, user=user, executor=executor)
        if executor.stats["created_schemas"] > 0 or executor.stats["evidence_added"] > 0:
            self._mark_clustered_processed(materials["clusters"])
        else:
            logger.info(
                "s2_react user=%s produced 0 schemas/evidence — skip marking %d facts as processed",
                user_id, sum(len(c["facts"]) for c in materials["clusters"]),
            )
        logger.info(
            "s2_react done user=%s schemas=%d intentions=%d evidence=%d edges=%d",
            user_id,
            executor.stats["created_schemas"],
            executor.stats["created_intentions"],
            executor.stats["evidence_added"],
            executor.stats["edges_added"],
        )
        return dict(executor.stats)

    async def _run_single_shot(
        self, *, system: str, user: str, executor: System2ToolExecutor
    ) -> None:
        """One chat_json call → parse ops array → dispatch each op via the executor.

        NOTE(dual_vs_hy): 对标 hy-memory ultra 的 single-shot JSON ops —— 输出裸 JSON
        数组（不是 {"ops": [...]} 对象）。4B 模型在 json_object 对象模式下长输出容易
        截断/漂移导致 parse 失败（实测 5996 字符 unparseable → 0 schemas）；裸数组
        配精确正则更鲁棒。
        """
        llm = self.factory.llm
        assert llm is not None, "System2Agent requires factory.llm"
        request_id = f"s2::{executor.app_id}::{executor.user_id}"
        _dump_dir = os.environ.get("DUAL_MEM_S2_DUMP_DIR", "")
        if _dump_dir:
            try:
                os.makedirs(_dump_dir, exist_ok=True)
                import hashlib
                _k = hashlib.md5(executor.user_id.encode()).hexdigest()[:10]
                with open(os.path.join(_dump_dir, f"s2_{_k}.txt"), "w") as _f:
                    _f.write("=== SYSTEM ===\n" + system + "\n\n=== USER ===\n" + user + "\n")
            except Exception:
                pass
        try:
            data = await llm.chat_json(system=system, user=user, json_array=True, max_tokens=16384)
        except json.JSONDecodeError as exc:
            logger.warning("[s2-single] llm JSON parse failed: %s", exc)
            self._log_trajectory(request_id, executor, 1, error=str(exc))
            return
        if _dump_dir:
            try:
                import hashlib
                _k = hashlib.md5(executor.user_id.encode()).hexdigest()[:10]
                with open(os.path.join(_dump_dir, f"s2_{_k}_out.json"), "w") as _f:
                    _f.write(str(data))
            except Exception:
                pass

        if isinstance(data, dict):
            ops = data.get("ops")
        else:
            ops = data
        if not isinstance(ops, list):
            ops = []
        logger.info(
            "s2_single user=%s raw_type=%s raw_len=%s ops=%d",
            executor.user_id, type(data).__name__, len(str(data)), len(ops),
        )
        for op in ops:
            if not isinstance(op, dict):
                continue
            name = str(op.get("op") or "").strip()
            if not name:
                continue
            arguments = {k: v for k, v in op.items() if k != "op"}
            await executor.execute(name=name, arguments=arguments)
        self._log_trajectory(request_id, executor, 1)

    def _log_trajectory(
        self, request_id: str, executor: System2ToolExecutor, iters: int, *, error: str | None = None
    ) -> None:
        """Persist the full single-shot ops log + final stats for replay/debug."""
        payload: dict = {
            "iters": iters,
            "stats": dict(executor.stats),
            "tool_call_log": list(executor.tool_call_log),
        }
        if error is not None:
            payload["error"] = error
        try:
            self.factory.cache.log_pipeline(
                request_id=request_id,
                stage="S2_AGENT_TRAJECTORY",
                payload=payload,
            )
        except Exception:
            pass

    def _mark_clustered_processed(self, clusters: list[dict]) -> None:
        """Mark clustered facts as processed so unused ones are not re-consumed next digest.

        Clustered facts all originate from ``prepare_materials`` where only nodes with
        ``s2_evidence_count == 0`` are selected, so we can patch them directly. This uses
        a single batched metadata update instead of a get+upsert per fact (the latter
        rewrote the embedding and triggered an HNSW/SQLite write for every node).
        """
        patches: dict[str, dict] = {}
        for cluster in clusters:
            for fact in cluster["facts"]:
                patches[fact["node_id"]] = {"s2_evidence_count": 1}
        if patches:
            self.factory.vector.update_payload_many(patches)
