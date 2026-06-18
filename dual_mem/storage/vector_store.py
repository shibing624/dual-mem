# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Vector store abstraction and a Chroma-backed implementation for
upserting, querying and managing MemoryNode embeddings with metadata filters.
"""
from abc import ABC, abstractmethod

import chromadb
from chromadb.config import Settings

from dual_mem.types import MemoryNode, MemoryStatus

_COLLECTION = "memories"


def _normalize_where(where: dict | None) -> dict | None:
    """Wrap a multi-key where-filter in Chroma's $and form; pass single keys through."""
    if not where:
        return None
    if len(where) == 1:
        return where
    return {"$and": [{key: value} for key, value in where.items()]}


class VectorStore(ABC):
    """Abstract interface for embedding-backed memory storage."""

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
    """Persistent Chroma-backed vector store using cosine distance."""

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
        """Insert or update the given nodes (embeddings, content and metadata)."""
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
        """Nearest-neighbor search under a filter; node.score is cosine similarity (1 - distance)."""
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
        """Fetch a single node (with embedding) by id, or None if absent."""
        result = self.collection.get(
            ids=[node_id], include=["metadatas", "documents", "embeddings"]
        )
        if not result["ids"]:
            return None
        embeddings = result["embeddings"]
        embedding = list(embeddings[0]) if embeddings is not None and len(embeddings) else None
        return MemoryNode.from_storage(result["documents"][0], result["metadatas"][0], embedding)

    def get_many(self, where: dict, limit: int = 1000) -> list[MemoryNode]:
        """Fetch all nodes matching a metadata filter (no embeddings), up to limit."""
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
        """Merge a metadata patch into an existing node; no-op if it does not exist."""
        current = self.collection.get(ids=[node_id], include=["metadatas"])
        if not current["ids"]:
            return
        meta = dict(current["metadatas"][0])
        meta.update(patch)
        self.collection.update(ids=[node_id], metadatas=[meta])

    def update_status(self, node_id: str, status: MemoryStatus) -> None:
        """Update only the status field of a node."""
        self.update_payload(node_id, {"status": status.value})

    def delete(self, node_ids: list[str]) -> None:
        """Physically remove the given node ids from the collection."""
        if not node_ids:
            return
        self.collection.delete(ids=node_ids)
