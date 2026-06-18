import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import kuzu

_REL_TYPES = {"RELATED_TO", "CROSS_ABSTRACTS_TO"}

_SCHEMA_DDL = [
    "CREATE NODE TABLE IF NOT EXISTS Memory("
    "node_id STRING, layer STRING, content STRING, app_id STRING, "
    "user_id STRING, agent_id STRING, embedding DOUBLE[], tags STRING[], "
    "gmt_created INT64, custom STRING, PRIMARY KEY(node_id))",
    "CREATE NODE TABLE IF NOT EXISTS VdbRef(node_id STRING, PRIMARY KEY(node_id))",
    "CREATE NODE TABLE IF NOT EXISTS Topic(name STRING, PRIMARY KEY(name))",
    "CREATE REL TABLE IF NOT EXISTS RELATED_TO(FROM Memory TO Memory)",
    "CREATE REL TABLE IF NOT EXISTS CROSS_ABSTRACTS_TO(FROM Memory TO Memory)",
    "CREATE REL TABLE IF NOT EXISTS DERIVED_FROM(FROM Memory TO VdbRef)",
    "CREATE REL TABLE IF NOT EXISTS TAGGED_WITH(FROM Memory TO Topic)",
]


@dataclass
class GraphNode:
    node_id: str
    layer: str
    content: str
    app_id: str
    user_id: str
    agent_id: str = ""
    embedding: list[float] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    gmt_created: int = 0
    custom: dict | None = None
    score: float = 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class GraphStore(ABC):
    @abstractmethod
    def add_node(self, node: GraphNode) -> None: ...

    @abstractmethod
    def query_by_embedding(
        self,
        *,
        layer: str,
        user_id: str,
        app_ids: list[str],
        embedding: list[float],
        top_k: int = 10,
    ) -> list[GraphNode]: ...

    @abstractmethod
    def list_by_layer(
        self, *, layer: str, user_id: str, app_ids: list[str], limit: int = 1000
    ) -> list[GraphNode]: ...

    @abstractmethod
    def add_evidence(self, *, schema_id: str, fact_id: str) -> None: ...

    @abstractmethod
    def evidence_of(self, schema_id: str) -> list[str]: ...

    @abstractmethod
    def add_edge(self, *, from_id: str, to_id: str, rel: str) -> None: ...

    @abstractmethod
    def neighbors_by_tag(
        self, *, tag: str, user_id: str, app_ids: list[str]
    ) -> list[str]: ...


class KuzuGraphStore(GraphStore):
    def __init__(self, storage_dir: str):
        self.db = kuzu.Database(f"{storage_dir}/kuzu")
        self.conn = kuzu.Connection(self.db)
        for ddl in _SCHEMA_DDL:
            self.conn.execute(ddl)

    def add_node(self, node: GraphNode) -> None:
        self.conn.execute(
            "MERGE (m:Memory {node_id: $nid}) "
            "SET m.layer=$layer, m.content=$content, m.app_id=$app_id, "
            "m.user_id=$user_id, m.agent_id=$agent_id, m.embedding=$embedding, "
            "m.tags=$tags, m.gmt_created=$gmt_created, m.custom=$custom",
            {
                "nid": node.node_id,
                "layer": node.layer,
                "content": node.content,
                "app_id": node.app_id,
                "user_id": node.user_id,
                "agent_id": node.agent_id,
                "embedding": node.embedding,
                "tags": node.tags,
                "gmt_created": node.gmt_created,
                "custom": json.dumps(node.custom or {}, ensure_ascii=False),
            },
        )
        for tag in node.tags:
            self.conn.execute("MERGE (t:Topic {name: $name})", {"name": tag})
            self.conn.execute(
                "MATCH (m:Memory {node_id: $nid}), (t:Topic {name: $name}) "
                "MERGE (m)-[:TAGGED_WITH]->(t)",
                {"nid": node.node_id, "name": tag},
            )

    def query_by_embedding(
        self,
        *,
        layer: str,
        user_id: str,
        app_ids: list[str],
        embedding: list[float],
        top_k: int = 10,
    ) -> list[GraphNode]:
        result = self.conn.execute(
            "MATCH (m:Memory) WHERE m.layer=$layer AND m.user_id=$user_id "
            "AND m.app_id IN $app_ids "
            "RETURN m.node_id, m.content, m.app_id, m.user_id, m.agent_id, "
            "m.embedding, m.tags, m.gmt_created, m.custom",
            {"layer": layer, "user_id": user_id, "app_ids": app_ids},
        )
        nodes = self._rows_to_nodes(result, layer)
        for node in nodes:
            node.score = _cosine(embedding, node.embedding)
        nodes.sort(key=lambda n: n.score, reverse=True)
        return nodes[:top_k]

    def list_by_layer(
        self, *, layer: str, user_id: str, app_ids: list[str], limit: int = 1000
    ) -> list[GraphNode]:
        result = self.conn.execute(
            "MATCH (m:Memory) WHERE m.layer=$layer AND m.user_id=$user_id "
            "AND m.app_id IN $app_ids "
            "RETURN m.node_id, m.content, m.app_id, m.user_id, m.agent_id, "
            "m.embedding, m.tags, m.gmt_created, m.custom "
            "ORDER BY m.gmt_created ASC LIMIT $limit",
            {"layer": layer, "user_id": user_id, "app_ids": app_ids, "limit": limit},
        )
        return self._rows_to_nodes(result, layer)

    @staticmethod
    def _rows_to_nodes(result, layer: str) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        while result.has_next():
            row = result.get_next()
            nodes.append(
                GraphNode(
                    node_id=row[0],
                    layer=layer,
                    content=row[1],
                    app_id=row[2],
                    user_id=row[3],
                    agent_id=row[4],
                    embedding=row[5],
                    tags=row[6],
                    gmt_created=row[7],
                    custom=json.loads(row[8]) or None if row[8] else None,
                )
            )
        return nodes

    def add_evidence(self, *, schema_id: str, fact_id: str) -> None:
        self.conn.execute("MERGE (v:VdbRef {node_id: $fid})", {"fid": fact_id})
        self.conn.execute(
            "MATCH (m:Memory {node_id: $sid}), (v:VdbRef {node_id: $fid}) "
            "MERGE (m)-[:DERIVED_FROM]->(v)",
            {"sid": schema_id, "fid": fact_id},
        )

    def evidence_of(self, schema_id: str) -> list[str]:
        result = self.conn.execute(
            "MATCH (m:Memory {node_id: $sid})-[:DERIVED_FROM]->(v:VdbRef) "
            "RETURN v.node_id",
            {"sid": schema_id},
        )
        ids: list[str] = []
        while result.has_next():
            ids.append(result.get_next()[0])
        return ids

    def add_edge(self, *, from_id: str, to_id: str, rel: str) -> None:
        if rel not in _REL_TYPES:
            raise ValueError(f"unsupported rel: {rel}")
        self.conn.execute(
            f"MATCH (a:Memory {{node_id: $from_id}}), (b:Memory {{node_id: $to_id}}) "
            f"MERGE (a)-[:{rel}]->(b)",
            {"from_id": from_id, "to_id": to_id},
        )

    def neighbors_by_tag(
        self, *, tag: str, user_id: str, app_ids: list[str]
    ) -> list[str]:
        result = self.conn.execute(
            "MATCH (m:Memory)-[:TAGGED_WITH]->(t:Topic {name: $tag}) "
            "WHERE m.user_id=$user_id AND m.app_id IN $app_ids "
            "RETURN m.node_id",
            {"tag": tag, "user_id": user_id, "app_ids": app_ids},
        )
        ids: list[str] = []
        while result.has_next():
            ids.append(result.get_next()[0])
        return ids
