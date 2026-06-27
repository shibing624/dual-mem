# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Vector store abstraction and a Chroma-backed implementation for
upserting, querying and managing MemoryNode embeddings with metadata filters.
"""
from abc import ABC, abstractmethod
import threading

import chromadb
from chromadb.config import Settings

from dual_mem.types import _LIST_SEP, MemoryNode, MemoryStatus

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
    def get_by_ids(self, node_ids: list[str]) -> dict[str, MemoryNode]: ...

    @abstractmethod
    def get_many(self, where: dict, limit: int = 1000) -> list[MemoryNode]: ...

    @abstractmethod
    def update_payload(self, node_id: str, patch: dict) -> None: ...

    @abstractmethod
    def update_status(self, node_id: str, status: MemoryStatus) -> None: ...

    @abstractmethod
    def mark_superseded(self, node_id: str, *, superseded_by_id: str) -> bool: ...

    @abstractmethod
    def delete(self, node_ids: list[str]) -> None: ...


class ChromaVectorStore(VectorStore):
    """Persistent Chroma-backed vector store using cosine distance."""

    def __init__(self, storage_dir: str):
        self._lock = threading.RLock()
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            result = self.collection.get(
                ids=[node_id], include=["metadatas", "documents", "embeddings"]
            )
        if not result["ids"]:
            return None
        embeddings = result["embeddings"]
        embedding = list(embeddings[0]) if embeddings is not None and len(embeddings) else None
        return MemoryNode.from_storage(result["documents"][0], result["metadatas"][0], embedding)

    def get_by_ids(self, node_ids: list[str]) -> dict[str, MemoryNode]:
        """Batch-fetch nodes by id (with embeddings when stored)."""
        if not node_ids:
            return {}
        unique_ids = list(dict.fromkeys(node_ids))
        with self._lock:
            result = self.collection.get(
                ids=unique_ids,
                include=["metadatas", "documents", "embeddings"],
            )
        out: dict[str, MemoryNode] = {}
        embeddings = result.get("embeddings")
        for i, nid in enumerate(result["ids"]):
            emb = None
            if embeddings is not None and i < len(embeddings) and embeddings[i] is not None:
                emb = list(embeddings[i])
            node = MemoryNode.from_storage(result["documents"][i], result["metadatas"][i], emb)
            out[nid] = node
        return out

    def get_many(self, where: dict, limit: int = 1000) -> list[MemoryNode]:
        """Fetch all nodes matching a metadata filter (no embeddings), up to limit."""
        with self._lock:
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
        with self._lock:
            current = self.collection.get(ids=[node_id], include=["metadatas"])
            if not current["ids"]:
                return
            meta = dict(current["metadatas"][0])
            meta.update(patch)
            self.collection.update(ids=[node_id], metadatas=[meta])

    def update_status(self, node_id: str, status: MemoryStatus) -> None:
        """Update only the status field of a node."""
        self.update_payload(node_id, {"status": status.value})

    def mark_superseded(self, node_id: str, *, superseded_by_id: str) -> bool:
        """Atomically flag a node as superseded by ``superseded_by_id``.

        Sets ``is_latest=False`` + ``status=SUPERSEDED`` and appends to ``superseded_by`` in a
        single locked read-modify-write on the metadata only (the embedding is never rewritten),
        so concurrent supersede operations on the same evolution chain cannot lose updates.
        Returns False if the node does not exist.
        """
        with self._lock:
            current = self.collection.get(ids=[node_id], include=["metadatas"])
            if not current["ids"]:
                return False
            meta = dict(current["metadatas"][0])
            existing = meta.get("superseded_by") or ""
            ids = existing.split(_LIST_SEP) if existing else []
            if superseded_by_id not in ids:
                ids.append(superseded_by_id)
            meta["superseded_by"] = _LIST_SEP.join(ids)
            meta["is_latest"] = False
            meta["status"] = MemoryStatus.SUPERSEDED.value
            self.collection.update(ids=[node_id], metadatas=[meta])
            return True

    def delete(self, node_ids: list[str]) -> None:
        """Physically remove the given node ids from the collection."""
        if not node_ids:
            return
        with self._lock:
            self.collection.delete(ids=node_ids)
