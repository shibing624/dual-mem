# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Executes System2 ops against the graph store: create_schema/create_intention
nodes, add_evidence edges (bumping fact evidence counts) and add_edge relations.
"""
import time
import uuid

from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
from dual_mem.types import Layer

_ALLOWED_RELS = {"RELATED_TO", "CROSS_ABSTRACTS_TO"}


def _str_list(value) -> list[str]:
    """Coerce a value into a list of non-empty strings (empty list otherwise)."""
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and v.strip()]


class ToolExecutor:
    """Applies System2-generated ops to the graph store and returns operation counts."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    def apply(self, *, ops: list[dict], app_id: str, user_id: str, agent_id: str) -> dict:
        """Execute a list of System2 ops sequentially; return counts of each effect."""
        graph = self.factory.graph
        created_schemas = 0
        created_intentions = 0
        evidence_added = 0
        edges_added = 0

        for op in ops:
            if not isinstance(op, dict):
                continue
            kind = op.get("op")
            if kind in ("create_schema", "create_intention"):
                content = op.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                layer = Layer.L6_SCHEMA if kind == "create_schema" else Layer.L7_INTENTION
                node_id = self._create_node(layer, op, app_id, user_id, agent_id)
                evidence_added += self._link_evidence(node_id, _str_list(op.get("evidence")))
                if kind == "create_schema":
                    created_schemas += 1
                else:
                    created_intentions += 1
            elif kind == "add_evidence":
                schema_id = op.get("schema_id")
                if not isinstance(schema_id, str) or not schema_id.strip():
                    continue
                evidence_added += self._link_evidence(schema_id, _str_list(op.get("evidence")))
            elif kind == "add_edge":
                from_id = op.get("from_id")
                to_id = op.get("to_id")
                if not isinstance(from_id, str) or not isinstance(to_id, str):
                    continue
                if not from_id.strip() or not to_id.strip():
                    continue
                rel = op.get("rel", "RELATED_TO")
                if rel not in _ALLOWED_RELS:
                    rel = "RELATED_TO"
                graph.add_edge(from_id=from_id, to_id=to_id, rel=rel)
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
        """Create and persist a new graph node for the given layer; return its id."""
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
        """Link fact evidence to a node and bump each fact's evidence count; return count added."""
        added = 0
        for fact_id in fact_ids:
            self.factory.graph.add_evidence(schema_id=schema_id, fact_id=fact_id)
            self._bump_evidence_count(fact_id)
            added += 1
        return added

    def _bump_evidence_count(self, fact_id: str) -> None:
        """Increment a fact node's System2 evidence count by one."""
        node = self.factory.vector.get(fact_id)
        if node is None:
            return
        node.s2_evidence_count += 1
        self.factory.vector.upsert([node])
