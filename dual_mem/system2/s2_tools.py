# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: System2 ReAct toolset. Defines the 8 OpenAI function-calling tools the LLM
agent can call (4 read tools, 4 write tools), plus a synchronous dispatcher (executor) that
runs each tool against the existing vector / graph stores. The dispatcher returns a string
result that becomes a ``role=tool`` message back to the LLM in the next ReAct turn.
"""
import json
import logging
import time
import uuid

from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
from dual_mem.types import Layer, MemoryStatus

logger = logging.getLogger("dual_mem.system2.tools")

_ALLOWED_RELS = {"RELATED_TO", "CROSS_ABSTRACTS_TO"}


# 8 OpenAI function tool definitions: 4 read + 4 write.
SYSTEM2_TOOL_DEFINITIONS: list[dict] = [
    # ---- Read tools ----------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "search_vdb",
            "description": (
                "Vector-database semantic search over user memories. Returns up to top_k "
                "active memories (L2/L4) closest to the query string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text to embed."},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_graph",
            "description": (
                "Search existing graph nodes (L6 schema or L7 intention) semantically. "
                "Use this BEFORE create_schema to avoid duplicates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "layer": {"type": "string", "enum": ["L6_SCHEMA", "L7_INTENTION"]},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query", "layer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_node",
            "description": "Fetch a single memory node by id (vector or graph store).",
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_node",
            "description": (
                "Return the 1-hop neighbours of a graph node (its evidence facts via "
                "DERIVED_FROM and direct RELATED_TO neighbours). Used to inspect what a "
                "schema is built on before adding more evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
    },
    # ---- Write tools ---------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "create_schema",
            "description": (
                "Create a new L6 schema describing ONE behavioural pattern in ONE domain. "
                "Content format: \"When [circumstance], the user [pattern] — reflecting "
                "[insight].\". Schema content is IMMUTABLE. Pass evidence fact ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_intention",
            "description": (
                "Create a new L7 intention: a CONCRETE future event/plan with clear "
                "action + temporal boundedness. NOT for life aspirations or values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_evidence",
            "description": (
                "Add one or more fact node ids as evidence for an EXISTING schema/intention. "
                "Use this instead of create_schema when a matching schema already exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "schema_id": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["schema_id", "evidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_edge",
            "description": (
                "Add a relation edge between two graph nodes. Allowed rels: "
                "RELATED_TO, CROSS_ABSTRACTS_TO. Default RELATED_TO."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_id": {"type": "string"},
                    "to_id": {"type": "string"},
                    "rel": {"type": "string"},
                },
                "required": ["from_id", "to_id"],
            },
        },
    },
]


def _ok(payload) -> str:
    return json.dumps({"ok": True, "result": payload}, ensure_ascii=False)


def _err(reason: str) -> str:
    return json.dumps({"ok": False, "error": reason}, ensure_ascii=False)


class System2ToolExecutor:
    """Dispatches one tool call against the configured stores; returns a JSON string for
    the ReAct ``role=tool`` reply. Maintains aggregate counters for the agent stats."""

    def __init__(self, *, factory: ComponentFactory, app_id: str, user_id: str, agent_id: str = ""):
        self.factory = factory
        self.app_id = app_id
        self.user_id = user_id
        self.agent_id = agent_id
        self.stats = {
            "created_schemas": 0,
            "created_intentions": 0,
            "evidence_added": 0,
            "edges_added": 0,
        }
        self.tool_call_log: list[dict] = []

    async def execute(self, *, name: str, arguments: dict) -> str:
        """Run one tool call and return a JSON string that the LLM consumes as role=tool."""
        try:
            if name == "search_vdb":
                result = await self._search_vdb(arguments)
            elif name == "search_graph":
                result = self._search_graph(arguments)
            elif name == "get_node":
                result = self._get_node(arguments)
            elif name == "expand_node":
                result = self._expand_node(arguments)
            elif name == "create_schema":
                result = await self._create_node(arguments, layer=Layer.L6_SCHEMA)
                if result.startswith('{"ok": true'):
                    self.stats["created_schemas"] += 1
            elif name == "create_intention":
                result = await self._create_node(arguments, layer=Layer.L7_INTENTION)
                if result.startswith('{"ok": true'):
                    self.stats["created_intentions"] += 1
            elif name == "add_evidence":
                result = self._add_evidence(arguments)
            elif name == "add_edge":
                result = self._add_edge(arguments)
            else:
                result = _err(f"unknown tool: {name}")
        except Exception as exc:
            result = _err(f"tool {name} raised: {exc}")
            logger.warning("[s2-tool] %s raised: %s", name, exc)
        self.tool_call_log.append({"tool": name, "args": arguments, "result": result})
        logger.debug("[s2-tool] %s args=%s ok=%s", name, arguments, result.startswith('{"ok": true'))
        return result

    # ---- Read tools ----------------------------------------------------------------

    async def _search_vdb(self, args: dict) -> str:
        query = (args.get("query") or "").strip()
        top_k = int(args.get("top_k") or 5)
        if not query:
            return _err("query is required")
        embedding = await self.factory.embed.embed(query)
        where = build_filter(
            app_ids=[self.app_id],
            user_id=self.user_id,
            agent_ids=[self.agent_id],
            layers=[Layer.L2_FACT, Layer.L4_IDENTITY],
            statuses=[MemoryStatus.ACTIVE],
        )
        nodes = self.factory.vector.query(embedding=embedding, where=where, top_k=top_k)
        return _ok(
            [
                {
                    "node_id": n.node_id,
                    "content": n.content,
                    "layer": n.layer.value,
                    "score": round(n.score, 4),
                }
                for n in nodes
            ]
        )

    def _search_graph(self, args: dict) -> str:
        graph = self.factory.graph
        if graph is None:
            return _err("graph store not enabled (dual mode only)")
        query = (args.get("query") or "").strip()
        layer = (args.get("layer") or Layer.L6_SCHEMA.value).strip()
        top_k = int(args.get("top_k") or 5)
        if not query:
            return _err("query is required")
        # Embed synchronously through embed_sync if available, else skip.
        embed = self.factory.embed
        embedding = getattr(embed, "embed_sync", None)
        if embedding is None:
            # Fallback: cheap deterministic vector via length (unit tests inject FakeEmbed which has embed_sync)
            return _err("embed_sync not available on this embed provider")
        emb = embedding(query)
        try:
            hits = graph.query_by_embedding(
                layer=layer, user_id=self.user_id, app_ids=[self.app_id],
                embedding=emb, top_k=top_k,
            )
        except Exception as exc:
            return _err(f"graph query failed: {exc}")
        return _ok(
            [
                {
                    "node_id": h.node_id,
                    "content": h.content,
                    "layer": h.layer,
                    "score": round(h.score, 4),
                }
                for h in hits
            ]
        )

    def _get_node(self, args: dict) -> str:
        nid = (args.get("node_id") or "").strip()
        if not nid:
            return _err("node_id is required")
        # Vector first, then graph fallback.
        node = self.factory.vector.get(nid)
        if node is not None:
            return _ok(
                {
                    "node_id": node.node_id,
                    "content": node.content,
                    "layer": node.layer.value,
                    "tags": list(node.tags),
                }
            )
        graph = self.factory.graph
        if graph is None:
            return _err(f"node {nid} not found")
        # No direct get_by_id on graph; iterate via list_by_layer is too expensive — return not found.
        return _err(f"node {nid} not found")

    def _expand_node(self, args: dict) -> str:
        nid = (args.get("node_id") or "").strip()
        if not nid:
            return _err("node_id is required")
        graph = self.factory.graph
        if graph is None:
            return _err("graph store not enabled")
        try:
            evidence = graph.evidence_of(nid)
        except Exception:
            evidence = []
        return _ok({"evidence_ids": evidence})

    # ---- Write tools ---------------------------------------------------------------

    async def _create_node(self, args: dict, *, layer: Layer) -> str:
        graph = self.factory.graph
        if graph is None:
            return _err("graph store not enabled")
        content = (args.get("content") or "").strip()
        if not content:
            return _err("content is required")
        node_id = str(uuid.uuid4())
        embedding = await self.factory.embed.embed(content)
        graph.add_node(
            GraphNode(
                node_id=node_id,
                layer=layer.value,
                content=content,
                app_id=self.app_id,
                user_id=self.user_id,
                agent_id=self.agent_id,
                embedding=embedding,
                tags=_str_list(args.get("tags")),
                gmt_created=int(time.time()),
            )
        )
        evidence = _str_list(args.get("evidence"))
        added = self._link_evidence(node_id, evidence)
        self.stats["evidence_added"] += added
        return _ok({"node_id": node_id, "layer": layer.value, "evidence_added": added})

    def _add_evidence(self, args: dict) -> str:
        graph = self.factory.graph
        if graph is None:
            return _err("graph store not enabled")
        schema_id = (args.get("schema_id") or "").strip()
        evidence = _str_list(args.get("evidence"))
        if not schema_id or not evidence:
            return _err("schema_id and evidence are required")
        added = self._link_evidence(schema_id, evidence)
        self.stats["evidence_added"] += added
        return _ok({"schema_id": schema_id, "evidence_added": added})

    def _add_edge(self, args: dict) -> str:
        graph = self.factory.graph
        if graph is None:
            return _err("graph store not enabled")
        from_id = (args.get("from_id") or "").strip()
        to_id = (args.get("to_id") or "").strip()
        if not from_id or not to_id:
            return _err("from_id and to_id are required")
        rel = (args.get("rel") or "RELATED_TO").strip()
        if rel not in _ALLOWED_RELS:
            rel = "RELATED_TO"
        try:
            graph.add_edge(from_id=from_id, to_id=to_id, rel=rel)
        except Exception as exc:
            return _err(f"add_edge failed: {exc}")
        self.stats["edges_added"] += 1
        return _ok({"from_id": from_id, "to_id": to_id, "rel": rel})

    def _link_evidence(self, node_id: str, fact_ids: list[str]) -> int:
        """Link facts as evidence + bump each fact's s2_evidence_count."""
        graph = self.factory.graph
        if graph is None or not fact_ids:
            return 0
        added = 0
        for fid in fact_ids:
            try:
                graph.add_evidence(schema_id=node_id, fact_id=fid)
            except Exception:
                continue
            self._bump_evidence_count(fid)
            added += 1
        return added

    def _bump_evidence_count(self, fact_id: str) -> None:
        node = self.factory.vector.get(fact_id)
        if node is None:
            return
        node.s2_evidence_count += 1
        self.factory.vector.upsert([node])


def _str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
