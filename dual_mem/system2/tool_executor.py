"""System2 ops 执行器。

把 LLM 输出的 ops JSON 数组串行落库到图谱：
- create_schema  → L6_SCHEMA 节点 + DERIVED_FROM 证据边 + fact 证据计数 +1
- create_intention → L7_INTENTION 节点（同样支持证据）
- add_evidence   → 给已有节点追加 DERIVED_FROM 证据边 + 计数 +1
- add_edge       → 两节点间建 RELATED_TO 关系
"""

import time
import uuid

from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
from dual_mem.types import Layer


class ToolExecutor:
    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    def apply(self, *, ops: list[dict], app_id: str, user_id: str, agent_id: str) -> dict:
        graph = self.factory.graph
        created_schemas = 0
        created_intentions = 0
        evidence_added = 0
        edges_added = 0

        for op in ops:
            kind = op.get("op")
            if kind == "create_schema":
                node_id = self._create_node(Layer.L6_SCHEMA, op, app_id, user_id, agent_id)
                evidence_added += self._link_evidence(node_id, op.get("evidence") or [])
                created_schemas += 1
            elif kind == "create_intention":
                node_id = self._create_node(Layer.L7_INTENTION, op, app_id, user_id, agent_id)
                evidence_added += self._link_evidence(node_id, op.get("evidence") or [])
                created_intentions += 1
            elif kind == "add_evidence":
                evidence_added += self._link_evidence(
                    op["schema_id"], op.get("evidence") or []
                )
            elif kind == "add_edge":
                graph.add_edge(
                    from_id=op["from_id"],
                    to_id=op["to_id"],
                    rel=op.get("rel", "RELATED_TO"),
                )
                edges_added += 1

        return {
            "created_schemas": created_schemas,
            "created_intentions": created_intentions,
            "evidence_added": evidence_added,
            "edges_added": edges_added,
        }

    def _create_node(
        self, layer: Layer, op: dict, app_id: str, user_id: str, agent_id: str
    ) -> str:
        node_id = str(uuid.uuid4())
        content = op.get("content", "")
        self.factory.graph.add_node(
            GraphNode(
                node_id=node_id,
                layer=layer.value,
                content=content,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                embedding=self.factory.embed.embed(content),
                tags=op.get("tags") or [],
                gmt_created=int(time.time()),
            )
        )
        return node_id

    def _link_evidence(self, schema_id: str, fact_ids: list[str]) -> int:
        added = 0
        for fact_id in fact_ids:
            self.factory.graph.add_evidence(schema_id=schema_id, fact_id=fact_id)
            self._bump_evidence_count(fact_id)
            added += 1
        return added

    def _bump_evidence_count(self, fact_id: str) -> None:
        node = self.factory.vector.get(fact_id)
        if node is None:
            return
        node.s2_evidence_count += 1
        self.factory.vector.upsert([node])
