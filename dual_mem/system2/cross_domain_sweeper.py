"""跨域升维扫描器（简化版）。

源码 ``cross_domain_sweeper.py`` 用「行为升维 embed → 矩阵碰撞 + Union-Find 聚类 →
LLM 突破性归纳」三部曲。我们按 plan 简化为：基础 L6 schema 数 >= 5 时触发，一次
LLM 调用归纳出一条核心 schema，并对涉及的基础 schema 建 CROSS_ABSTRACTS_TO 边。

核心 schema 用 GraphNode.custom={"sub_type": "core"} 标记，与基础 schema 区分。
"""

import time
import uuid

from dual_mem.providers.llm import is_chinese
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
from dual_mem.types import Layer

_MIN_BASICS = 5

CROSS_DOMAIN_PROMPT_ZH = """你是深层模式分析师。系统在用户不同生活领域的行为 Schema 间发现了结构性共鸣：表面看不相关，但底层行为逻辑高度相似。

请综合出一个**更高阶**的核心模式，解释这些行为为何共现（用户自己未必察觉）：聚焦深层认知风格、核心心理需求、隐含心智模型。只有当连接真正成立时才输出。

## 基础 Schema 列表
{patterns}

只输出一个 JSON 对象，不要任何额外文字：
{{"content": "一句话描述更高阶的核心模式", "schema_ids": ["参与归纳的基础 schema_id", ...]}}"""

CROSS_DOMAIN_PROMPT_EN = """You are a deep pattern analyst. The system has detected structural resonance among behavioral Schemas from different areas of the user's life: superficially unrelated, but their underlying behavioral logic is strikingly similar.

Synthesize ONE HIGHER-ORDER core pattern explaining why these behaviors co-occur (the user may not be consciously aware): focus on deep cognitive style, core psychological need, hidden mental model. Only output if the connection is genuinely compelling.

## Basic Schema list
{patterns}

Output ONLY a JSON object, nothing else:
{{"content": "one sentence describing the higher-order core pattern", "schema_ids": ["basic schema_id involved", ...]}}"""


class CrossDomainSweeper:
    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    def run(self, *, app_id: str, user_id: str, agent_id: str = "") -> dict:
        graph = self.factory.graph
        all_schemas = graph.list_by_layer(
            layer=Layer.L6_SCHEMA.value, user_id=user_id, app_ids=[app_id]
        )
        basics = [n for n in all_schemas if (n.custom or {}).get("sub_type") != "core"]

        if len(basics) < _MIN_BASICS:
            return {"triggered": False, "basics_count": len(basics)}

        patterns = "\n".join(f"- {n.content}  (id={n.node_id})" for n in basics)
        text = " ".join(n.content for n in basics)
        system = CROSS_DOMAIN_PROMPT_ZH if is_chinese(text) else CROSS_DOMAIN_PROMPT_EN
        parsed = self.factory.llm.chat_json(
            system=system, user=patterns
        )

        content = (parsed or {}).get("content", "")
        if not content:
            return {"triggered": True, "core_id": None, "abstracted": 0}

        basic_ids = {n.node_id for n in basics}
        targets = [sid for sid in (parsed.get("schema_ids") or []) if sid in basic_ids]
        if not targets:
            targets = list(basic_ids)

        core_id = str(uuid.uuid4())
        graph.add_node(
            GraphNode(
                node_id=core_id,
                layer=Layer.L6_SCHEMA.value,
                content=content,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                embedding=self.factory.embed.embed(content),
                custom={"sub_type": "core"},
                gmt_created=int(time.time()),
            )
        )
        for basic_id in targets:
            graph.add_edge(from_id=basic_id, to_id=core_id, rel="CROSS_ABSTRACTS_TO")

        return {"triggered": True, "core_id": core_id, "abstracted": len(targets)}
