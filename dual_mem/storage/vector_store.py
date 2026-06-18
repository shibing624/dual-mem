from abc import ABC, abstractmethod

import chromadb
from chromadb.config import Settings

from dual_mem.types import MemoryNode, MemoryStatus

_COLLECTION = "memories"


def _normalize_where(where: dict | None) -> dict | None:
    if not where:
        return None
    if len(where) == 1:
        return where
    return {"$and": [{key: value} for key, value in where.items()]}


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, nodes: list[MemoryNode]) -> None: ...

    @abstractmethod
    def query(
        self, *, embedding: list[float], where: dict, top_k: int = 10
    ) -> list[MemoryNode]: ...

    @abstractmethod
    def get(self, node_id: str) -> MemoryNode | None: ...

    @abstractmethod
    def get_many(self, where: dict, limit: int = 1000) -> list[MemoryNode]: ...

    @abstractmethod
    def update_payload(self, node_id: str, patch: dict) -> None: ...

    @abstractmethod
    def update_status(self, node_id: str, status: MemoryStatus) -> None: ...

    @abstractmethod
    def delete(self, node_ids: list[str]) -> None: ...


class ChromaVectorStore(VectorStore):
    def __init__(self, storage_dir: str):
        self.client = chromadb.PersistentClient(
            path=f"{storage_dir}/chroma",
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, nodes: list[MemoryNode]) -> None:
        if not nodes:
            return
        self.collection.upsert(
            ids=[n.node_id for n in nodes],
            embeddings=[n.embedding for n in nodes],
            documents=[n.content for n in nodes],
            metadatas=[n.to_metadata() for n in nodes],
        )

    def query(
        self, *, embedding: list[float], where: dict, top_k: int = 10
    ) -> list[MemoryNode]:
        result = self.collection.query(
            query_embeddings=[embedding],
            where=_normalize_where(where),
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        nodes: list[MemoryNode] = []
        for content, meta, dist in zip(documents, metadatas, distances):
            node = MemoryNode.from_storage(content, meta)
            node.score = 1.0 - dist
            nodes.append(node)
        return nodes

    def get(self, node_id: str) -> MemoryNode | None:
        result = self.collection.get(
            ids=[node_id], include=["metadatas", "documents", "embeddings"]
        )
        if not result["ids"]:
            return None
        embeddings = result["embeddings"]
        embedding = list(embeddings[0]) if embeddings is not None and len(embeddings) else None
        return MemoryNode.from_storage(result["documents"][0], result["metadatas"][0], embedding)

    def get_many(self, where: dict, limit: int = 1000) -> list[MemoryNode]:
        result = self.collection.get(
            where=_normalize_where(where),
            include=["metadatas", "documents"],
            limit=limit,
        )
        return [
            MemoryNode.from_storage(content, meta)
            for content, meta in zip(result["documents"], result["metadatas"])
        ]

    def update_payload(self, node_id: str, patch: dict) -> None:
        current = self.collection.get(ids=[node_id], include=["metadatas"])
        if not current["ids"]:
            return
        meta = dict(current["metadatas"][0])
        meta.update(patch)
        self.collection.update(ids=[node_id], metadatas=[meta])

    def update_status(self, node_id: str, status: MemoryStatus) -> None:
        self.update_payload(node_id, {"status": status.value})

    def delete(self, node_ids: list[str]) -> None:
        if not node_ids:
            return
        self.collection.delete(ids=node_ids)
