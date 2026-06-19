# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: System2 ReAct cognitive-processing agent. Phase 1 prepares clustered fact
materials (no LLM); phase 2 hands them to the LLM, which drives a true OpenAI
function-calling tool loop (search_vdb / search_graph / get_node / expand_node +
create_schema / create_intention / add_evidence / add_edge) until it emits no more
tool_calls. The legacy single-shot ops-JSON path has been deleted.
"""
import json
import logging

from dual_mem.agent import prompts
from dual_mem.isolation import build_filter
from dual_mem.providers.llm import is_chinese
from dual_mem.registry import ComponentFactory
from dual_mem.system2.clustering import cluster_facts
from dual_mem.system2.s2_tools import SYSTEM2_TOOL_DEFINITIONS, System2ToolExecutor
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

    fresh = []
    for fact in all_facts:
        if fact.s2_evidence_count == 0:
            full = factory.vector.get(fact.node_id)
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

    return {
        "clusters": clusters,
        "graph_forward": graph_forward,
        "existing_tags": existing_tags,
        "stats": {
            "total_facts": len(all_facts),
            "fresh_facts": len(fresh),
            "clusters_found": len(clusters),
            "existing_schemas": len(graph_forward),
        },
    }


def _build_user_message(materials: dict) -> str:
    """Render prepared materials (clusters, existing schemas, tags) into the user prompt."""
    parts = ["## 聚类结果 / Cluster results"]
    for i, cluster in enumerate(materials["clusters"]):
        parts.append(f"### Cluster {i}（主题: {cluster['centroid_text']}）")
        for fact in cluster["facts"]:
            parts.append(f"  - [{fact['layer']}] {fact['content']}  (id={fact['node_id']})")

    forward = materials["graph_forward"]
    if forward:
        parts.append("\n## 已有 Schema / Existing schemas")
        for node in forward:
            parts.append(f"  - {node['content']}  (id={node['node_id']})")

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
        user = _build_user_message(materials)

        executor = System2ToolExecutor(
            factory=self.factory, app_id=app_id, user_id=user_id, agent_id=agent_id
        )
        await self._run_react_loop(system=system, user=user, executor=executor)
        self._mark_clustered_processed(materials["clusters"])
        logger.info(
            "s2_react done user=%s schemas=%d intentions=%d evidence=%d edges=%d",
            user_id,
            executor.stats["created_schemas"],
            executor.stats["created_intentions"],
            executor.stats["evidence_added"],
            executor.stats["edges_added"],
        )
        return dict(executor.stats)

    async def _run_react_loop(
        self, *, system: str, user: str, executor: System2ToolExecutor
    ) -> None:
        """Drive the OpenAI function-calling loop until the LLM emits no more tool_calls."""
        llm = self.factory.llm
        assert llm is not None, "System2Agent requires factory.llm"

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        max_iters = max(1, self.factory.settings.system2_max_iters)
        request_id = f"s2::{executor.app_id}::{executor.user_id}"
        iters_run = 0
        for turn_idx in range(max_iters):
            iters_run = turn_idx + 1
            try:
                turn = await llm.chat_with_tools(
                    messages=messages,
                    tools=SYSTEM2_TOOL_DEFINITIONS,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.warning("[s2-react] llm call failed: %s", exc)
                self._log_trajectory(request_id, executor, iters_run, error=str(exc))
                return

            tool_calls = turn.get("tool_calls") or []
            assistant_msg: dict = {"role": "assistant", "content": turn.get("content") or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            try:
                self.factory.cache.log_pipeline(
                    request_id=request_id,
                    stage="S2_AGENT_TURN",
                    payload={
                        "turn": turn_idx,
                        "tool_call_count": len(tool_calls),
                        "tools": [(tc.get("function") or {}).get("name", "") for tc in tool_calls],
                    },
                )
            except Exception:
                pass

            if not tool_calls:
                break

            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                result_str = await executor.execute(name=name, arguments=args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id") or f"call_{name}",
                        "content": result_str,
                    }
                )

        self._log_trajectory(request_id, executor, iters_run)

    def _log_trajectory(
        self, request_id: str, executor: System2ToolExecutor, iters: int, *, error: str | None = None
    ) -> None:
        """Persist the full ReAct tool_call_log + final stats for replay/debug."""
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
        """Mark clustered facts as processed so unused ones are not re-consumed next digest."""
        for cluster in clusters:
            for fact in cluster["facts"]:
                node = self.factory.vector.get(fact["node_id"])
                if node is not None and node.s2_evidence_count == 0:
                    node.s2_evidence_count = 1
                    self.factory.vector.upsert([node])
