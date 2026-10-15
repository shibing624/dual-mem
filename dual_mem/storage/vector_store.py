# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Vector store abstraction and a Chroma-backed implementation for
upserting, querying and managing MemoryNode embeddings with metadata filters.
"""
from abc import ABC, abstractmethod
import threading
import json
import sqlite3
import hashlib
import math
import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import chromadb
from chromadb.config import Settings

from dual_mem.storage.history_store import HistoryStore
from dual_mem.types import _LIST_SEP, Layer, MemoryNode, MemoryStatus

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
    def update_payload_many(self, patches: dict[str, dict]) -> None: ...

    @abstractmethod
    def update_status(self, node_id: str, status: MemoryStatus) -> None: ...

    @abstractmethod
    def mark_superseded(
        self, node_id: str, *, superseded_by_id: str, keep_active: bool = False
    ) -> bool: ...

    @abstractmethod
    def link_supersedes(
        self, *, new_id: str, old_id: str, keep_active: bool = False
    ) -> bool: ...

    @abstractmethod
    def delete(self, node_ids: list[str]) -> None: ...


class ChromaVectorStore(VectorStore):
    """Persistent Chroma-backed vector store using cosine distance."""

    def __init__(self, storage_dir: str, *, history: HistoryStore | None = None):
        self._history = history
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._local = threading.local()
        self._lock_path = Path(storage_dir) / "versions.lock"
        self._journal = sqlite3.connect(
            str(Path(storage_dir) / "versions.db"),
            check_same_thread=False,
        )
        self._journal.execute(
            "CREATE TABLE IF NOT EXISTS epochs (scope TEXT PRIMARY KEY, epoch INTEGER NOT NULL)"
        )
        self._journal.execute(
            "CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, scope TEXT NOT NULL, epoch INTEGER NOT NULL, payload TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0)"
        )
        self._journal.execute(
            "CREATE TABLE IF NOT EXISTS projections (plan_id TEXT PRIMARY KEY, done INTEGER NOT NULL DEFAULT 0)"
        )
        self._journal.commit()
        self.client = chromadb.PersistentClient(
            path=f"{storage_dir}/chroma",
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        with self._guard():
            self._migrate_layers()

    @contextmanager
    def _guard(self) -> Iterator[None]:
        """Serialize cooperating processes and recover before exposing any read."""
        with self._lock:
            if getattr(self._local, "inside", False):
                yield
                return
            with self._lock_path.open("a") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                self._local.inside = True
                try:
                    for plan_id, scope_key, epoch, payload in self._journal.execute(
                        "SELECT id, scope, epoch, payload FROM plans WHERE done=0 ORDER BY rowid"
                    ).fetchall():
                        self._finish_plan(
                            plan_id, scope_key, epoch, json.loads(payload)
                        )
                    yield
                finally:
                    self._local.inside = False
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _finish_plan(
        self, plan_id: str, scope_key: str, epoch: int, payload: dict
    ) -> None:
        rows = payload["nodes"]
        cap = self.client.get_max_batch_size()
        for offset in range(0, len(rows), cap):
            chunk = rows[offset : offset + cap]
            self.collection.upsert(
                ids=[n["id"] for n in chunk],
                documents=[n["content"] for n in chunk],
                embeddings=[n["embedding"] for n in chunk],
                metadatas=[n["meta"] for n in chunk],
            )
        patches = payload["patches"]
        patch_ids = list(patches)
        for offset in range(0, len(patch_ids), cap):
            chunk_ids = patch_ids[offset : offset + cap]
            self.collection.update(
                ids=chunk_ids, metadatas=[patches[nid] for nid in chunk_ids]
            )
        if self._history:
            for i, event in enumerate(payload.get("audit", [])):
                self._history.append(**event, event_key=f"{plan_id}:{i}")
        with self._journal:
            if any(n["meta"]["layer"] == Layer.L3_DERIVED.value for n in rows):
                self._journal.execute(
                    "INSERT OR IGNORE INTO projections(plan_id) VALUES (?)", (plan_id,)
                )
            self._journal.execute(
                "INSERT INTO epochs(scope,epoch) VALUES (?,?) ON CONFLICT(scope) DO UPDATE SET epoch=excluded.epoch",
                (scope_key, epoch + 1),
            )
            self._journal.execute("UPDATE plans SET done=1 WHERE id=?", (plan_id,))

    def scope_epoch(self, scope: tuple[str, str, str]) -> int:
        with self._guard():
            row = self._journal.execute(
                "SELECT epoch FROM epochs WHERE scope=?", (json.dumps(scope),)
            ).fetchone()
            return row[0] if row else 0

    def _bump_epoch(self, scope: tuple[str, str, str]) -> None:
        with self._journal:
            self._journal.execute(
                "INSERT INTO epochs(scope,epoch) VALUES (?,1) ON CONFLICT(scope) DO UPDATE SET epoch=epoch+1",
                (json.dumps(scope),),
            )

    def _migrate_layers(self) -> None:
        """Migrate legacy metadata without touching embeddings or documents."""
        all_meta: dict[str, dict] = {}
        offset = 0
        while True:
            rows = self.collection.get(limit=500, offset=offset, include=["metadatas"])
            if not rows["ids"]:
                break
            all_meta.update(zip(rows["ids"], rows["metadatas"]))
            offset += len(rows["ids"])
        if all(m.get("layer_schema_version") == 4 for m in all_meta.values()):
            return

        def links(meta: dict, key: str) -> list[str]:
            return [x for x in (meta.get(key) or "").split(_LIST_SEP) if x]

        # Validate the entire chain before certifying any migrated row. Recover a
        # legacy append-before-link crash by restoring the missing reciprocal link.
        parents = {nid: links(meta, "supersedes") for nid, meta in all_meta.items()}
        successors = {
            nid: links(meta, "superseded_by") for nid, meta in all_meta.items()
        }
        for nid, meta in all_meta.items():
            scope = tuple(meta[k] for k in ("app_id", "user_id", "agent_id"))
            for target in [*parents[nid], *successors[nid]]:
                other = all_meta.get(target)
                if (
                    other is None
                    or tuple(other[k] for k in ("app_id", "user_id", "agent_id"))
                    != scope
                ):
                    raise ValueError(
                        "broken legacy evolution link; migration requires repair"
                    )
            for target in successors[nid]:
                if nid not in parents[target]:
                    raise ValueError(
                        "broken legacy evolution link; migration requires repair"
                    )
            for parent in parents[nid]:
                if nid not in successors[parent]:
                    successors[parent].append(nid)
        pending = {nid: len(set(p)) for nid, p in parents.items()}
        roots = [nid for nid, count in pending.items() if count == 0]
        visited = 0
        while roots:
            nid = roots.pop()
            visited += 1
            for child in successors[nid]:
                pending[child] -= 1
                if pending[child] == 0:
                    roots.append(child)
        if visited != len(all_meta):
            raise ValueError("cyclic legacy evolution link; migration requires repair")
        migrated = {}
        for nid, original in all_meta.items():
            meta = dict(original)
            old_layer = meta["layer"]
            meta["layer"] = Layer(old_layer).value
            meta["kind"] = {
                "L7_INTENTION": "intention",
                "L3_SUMMARY": "summary",
                "L6_SCHEMA": "schema",
                "L0_BASIC_INFO": "profile",
                "L4_IDENTITY": "profile",
            }.get(old_layer, meta.get("kind", ""))
            meta["layer_schema_version"] = 4
            meta["superseded_by"] = _LIST_SEP.join(successors[nid])
            if successors[nid]:
                meta["is_latest"] = False
            meta.setdefault("revision", 0)
            if meta.get("status") == MemoryStatus.SUPERSEDED.value:
                meta["status"] = MemoryStatus.ACTIVE.value
                meta["is_latest"] = False
            elif meta.get("status") in (
                MemoryStatus.SHADOW.value,
                MemoryStatus.DELETED.value,
            ):
                meta["status"] = MemoryStatus.ACTIVE.value
                meta["is_latest"] = False
                custom = json.loads(meta.get("custom") or "{}")
                custom["archived"] = True
                meta["custom"] = json.dumps(custom)
            migrated[nid] = meta
        ids = list(migrated)
        cap = min(500, self.client.get_max_batch_size())
        for offset in range(0, len(ids), cap):
            chunk = ids[offset : offset + cap]
            self.collection.update(
                ids=chunk, metadatas=[migrated[nid] for nid in chunk]
            )

    def upsert(self, nodes: list[MemoryNode]) -> None:
        """Insert or update the given nodes (embeddings, content and metadata)."""
        if not nodes:
            return
        with self._guard():
            old_nodes = self.get_by_ids([n.node_id for n in nodes])
            for node in nodes:
                old = old_nodes.get(node.node_id)
                if old:
                    if (old.app_id, old.user_id, old.agent_id) != (
                        node.app_id,
                        node.user_id,
                        node.agent_id,
                    ):
                        raise ValueError("cannot change node tenant")
                    if old.content != node.content:
                        raise ValueError(
                            "stored documents are immutable; append a version"
                        )
                    # An import/raw retry cannot replace a newer version's metadata.
                    # Documents are immutable; patches and version plans own mutations.
            nodes = [n for n in nodes if n.node_id not in old_nodes]
            if not nodes:
                return
            self.collection.upsert(
                ids=[n.node_id for n in nodes],
                embeddings=[n.embedding for n in nodes],
                documents=[n.content for n in nodes],
                metadatas=[n.to_metadata() for n in nodes],
            )
            for scope in {
                (n.app_id, n.user_id, n.agent_id)
                for n in nodes
                if n.layer != Layer.L0_RAW
            }:
                self._bump_epoch(scope)

    def query(
        self, *, embedding: list[float], where: dict, top_k: int = 10
    ) -> list[MemoryNode]:
        """Nearest-neighbor search under a filter; node.score is cosine similarity (1 - distance)."""
        with self._guard():
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
        with self._guard():
            result = self.collection.get(
                ids=[node_id], include=["metadatas", "documents", "embeddings"]
            )
        if not result["ids"]:
            return None
        embeddings = result["embeddings"]
        embedding = (
            list(embeddings[0]) if embeddings is not None and len(embeddings) else None
        )
        return MemoryNode.from_storage(
            result["documents"][0], result["metadatas"][0], embedding
        )

    def get_by_ids(self, node_ids: list[str]) -> dict[str, MemoryNode]:
        """Batch-fetch nodes by id (with embeddings when stored)."""
        if not node_ids:
            return {}
        unique_ids = list(dict.fromkeys(node_ids))
        with self._guard():
            result = self.collection.get(
                ids=unique_ids,
                include=["metadatas", "documents", "embeddings"],
            )
        out: dict[str, MemoryNode] = {}
        embeddings = result.get("embeddings")
        for i, nid in enumerate(result["ids"]):
            emb = None
            if (
                embeddings is not None
                and i < len(embeddings)
                and embeddings[i] is not None
            ):
                emb = list(embeddings[i])
            node = MemoryNode.from_storage(
                result["documents"][i], result["metadatas"][i], emb
            )
            out[nid] = node
        return out

    def get_many(self, where: dict, limit: int = 1000) -> list[MemoryNode]:
        """Fetch all nodes matching a metadata filter (no embeddings), up to limit."""
        with self._guard():
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
        with self._guard():
            current = self.collection.get(ids=[node_id], include=["metadatas"])
            if not current["ids"]:
                return
            meta = dict(current["metadatas"][0])
            meta.update(patch)
            if any(
                key in patch
                for key in (
                    "content",
                    "layer",
                    "status",
                    "is_latest",
                    "supersedes",
                    "superseded_by",
                    "custom",
                    "valid_from",
                    "valid_until",
                )
            ):
                meta["revision"] = int(current["metadatas"][0].get("revision", 0)) + 1
                self._bump_epoch((meta["app_id"], meta["user_id"], meta["agent_id"]))
            self.collection.update(ids=[node_id], metadatas=[meta])

    def update_payload_many(self, patches: dict[str, dict]) -> None:
        """Merge metadata patches into many existing nodes in two batched round-trips.

        Reads all target ids in one ``get`` (metadata only, no embeddings) and writes
        all merged metadatas in one ``update``. Embeddings are never rewritten. Ids that
        do not exist are skipped. This replaces per-node get/upsert loops which otherwise
        trigger one HNSW/SQLite write per node.
        """
        if not patches:
            return
        ids = list(patches.keys())
        with self._guard():
            current = self.collection.get(ids=ids, include=["metadatas"])
            existing_ids = current["ids"]
            if not existing_ids:
                return
            metas = current["metadatas"]
            new_ids: list[str] = []
            new_metas: list[dict] = []
            for nid, meta in zip(existing_ids, metas):
                merged = dict(meta)
                merged.update(patches[nid])
                if any(
                    key in patches[nid]
                    for key in (
                        "layer",
                        "status",
                        "is_latest",
                        "supersedes",
                        "superseded_by",
                        "custom",
                        "valid_from",
                        "valid_until",
                    )
                ):
                    merged["revision"] = int(meta.get("revision", 0)) + 1
                    self._bump_epoch(
                        (meta["app_id"], meta["user_id"], meta["agent_id"])
                    )
                new_ids.append(nid)
                new_metas.append(merged)
            self.collection.update(ids=new_ids, metadatas=new_metas)

    def update_status(self, node_id: str, status: MemoryStatus) -> None:
        """Update only the status field of a node."""
        self.update_payload(node_id, {"status": status.value})

    def mark_superseded(
        self, node_id: str, *, superseded_by_id: str, keep_active: bool = False
    ) -> bool:
        """Compatibility alias: link the retained ancestor and retire its head."""
        return self.link_supersedes(new_id=superseded_by_id, old_id=node_id)

    def link_supersedes(
        self, *, new_id: str, old_id: str, keep_active: bool = False
    ) -> bool:
        """Link retained versions; legacy keep_active no longer changes visibility."""
        if new_id == old_id:
            raise ValueError("a version cannot supersede itself")
        with self._guard():
            nodes = self.get_by_ids([new_id, old_id])
            if set(nodes) != {new_id, old_id}:
                return False
            new, old = nodes[new_id], nodes[old_id]
            scope = (new.app_id, new.user_id, new.agent_id)
            if scope != (old.app_id, old.user_id, old.agent_id):
                raise ValueError("version link crosses tenant scope")
            if old_id in new.supersedes and new_id in old.superseded_by:
                return True
            if not new.is_latest or not old.is_latest:
                raise ValueError("link requires current versions")
            pending, seen = [old_id], set()
            while pending:
                nid = pending.pop()
                if nid == new_id:
                    raise ValueError("cyclic version link")
                if nid in seen:
                    continue
                seen.add(nid)
                ancestor = self.get(nid)
                if ancestor:
                    pending.extend(ancestor.supersedes)
            new.supersedes = list(dict.fromkeys([*new.supersedes, old_id]))
            old.superseded_by = list(dict.fromkeys([*old.superseded_by, new_id]))
            old.is_latest = False
            old.status = MemoryStatus.ACTIVE
            new.revision += 1
            old.revision += 1
            self.collection.update(
                ids=[new_id, old_id], metadatas=[new.to_metadata(), old.to_metadata()]
            )
            self._bump_epoch(scope)
            return True

    def delete(self, node_ids: list[str]) -> None:
        """Physically remove the given node ids from the collection."""
        if not node_ids:
            return
        with self._guard():
            old = self.get_by_ids(node_ids)
            self.collection.delete(ids=node_ids)
            for scope in {(n.app_id, n.user_id, n.agent_id) for n in old.values()}:
                self._bump_epoch(scope)

    def close(self) -> None:
        with self._lock:
            self._journal.close()

    def completed_request(self, request_key: str) -> dict | None:
        """Recover a write result when the client crashed before its result cache."""
        with self._guard():
            for (payload,) in self._journal.execute(
                "SELECT payload FROM plans WHERE done=1 ORDER BY rowid DESC"
            ):
                plan = json.loads(payload)
                if plan.get("request_key") == request_key:
                    return {
                        "extra_node_ids": [n["id"] for n in plan["nodes"]],
                        "commit_passed": True,
                        "decisions": plan.get("decisions", []),
                    }
        return None

    def pending_projections(
        self, scope: tuple[str, str, str]
    ) -> list[tuple[str, list[str]]]:
        """Committed derived versions awaiting idempotent graph projection."""
        with self._guard():
            rows = self._journal.execute(
                "SELECT plans.id, plans.payload FROM plans JOIN projections ON plans.id=projections.plan_id WHERE plans.done=1 AND projections.done=0 AND plans.scope=? ORDER BY plans.rowid",
                (json.dumps(scope),),
            ).fetchall()
            return [
                (
                    pid,
                    [
                        n["id"]
                        for n in json.loads(payload)["nodes"]
                        if n["meta"]["layer"] == Layer.L3_DERIVED.value
                    ],
                )
                for pid, payload in rows
            ]

    def acknowledge_projection(self, plan_id: str) -> None:
        with self._guard(), self._journal:
            self._journal.execute(
                "UPDATE projections SET done=1 WHERE plan_id=?", (plan_id,)
            )

    def apply_version_plan(
        self,
        *,
        nodes: list[MemoryNode],
        expected: dict[str, int],
        archives: list[str],
        scope: tuple[str, str, str],
        expected_epoch: int | None = None,
        request_key: str | None = None,
        processed_evidence: list[str] | None = None,
        decisions: list[dict] | None = None,
    ) -> bool:
        """Recoverable append/link with revision and scope CAS, never erase documents.

        A durable journal holds the entire validated operation before Chroma is
        changed. Every store read recovers unfinished work under a process lock.
        Replaying a committed stable operation is a no-op; reusing an ID with a
        different document is rejected. Direct upsert is reserved for raw/import.
        """
        with self._guard():
            scope_key = json.dumps(scope)
            epoch = self.scope_epoch(scope)
            plan_key = json.dumps(
                [
                    scope,
                    [n.node_id for n in nodes],
                    sorted(archives),
                    sorted(expected.items()),
                    sorted(processed_evidence or []),
                ]
            )
            plan_id = hashlib.sha256(plan_key.encode()).hexdigest()
            replay = self._journal.execute(
                "SELECT payload FROM plans WHERE id=? AND done=1", (plan_id,)
            ).fetchone()
            if replay:
                previous = json.loads(replay[0])
                semantic = [
                    {
                        "id": n.node_id,
                        "content": n.content,
                        "layer": n.layer.value,
                        "kind": n.kind,
                        "supersedes": n.supersedes,
                        "source_ids": (n.custom or {}).get("source_ids", []),
                        "valid_from": n.valid_from,
                        "valid_until": n.valid_until,
                        "update_type": (n.custom or {}).get("update_type"),
                        "conflicts_with": (n.custom or {}).get("conflicts_with", []),
                        "owner": n.owner,
                        "session_id": n.session_id,
                        "memory_at": n.memory_at,
                        "tags": n.tags,
                        "temporal_scope": (n.custom or {}).get("temporal_scope"),
                        "negation": (n.custom or {}).get("negation", False),
                    }
                    for n in nodes
                ]
                if previous.get("semantic") != semantic:
                    raise ValueError(
                        "stable operation ID reused with different semantics"
                    )
                stored = self.get_by_ids([n.node_id for n in nodes])
                if any(
                    n.node_id not in stored or stored[n.node_id].content != n.content
                    for n in nodes
                ):
                    raise ValueError(
                        "stable operation ID reused with different content"
                    )
                return True
            if expected_epoch is not None and epoch != expected_epoch:
                return False
            current = self.get_by_ids(list(expected))
            if any(
                nid not in current
                or (current[nid].app_id, current[nid].user_id, current[nid].agent_id)
                != scope
                or current[nid].revision != rev
                or not current[nid].is_latest
                or current[nid].status != MemoryStatus.ACTIVE
                for nid, rev in expected.items()
            ):
                return False
            new_ids = {n.node_id for n in nodes}
            if len(new_ids) != len(nodes) or new_ids.intersection(current):
                raise ValueError(
                    "new version IDs must be unique and distinct from targets"
                )
            if self.get_by_ids(list(new_ids)):
                raise ValueError("new version IDs already exist")
            # Chroma retains dimension even after deleting its last document.
            # Fetch the current model: the original Collection's model is cached.
            expected_dim = (
                self.client.get_collection(name=_COLLECTION).get_model().dimension
            )
            if nodes:
                expected_dim = expected_dim or len(nodes[0].embedding or [])
            for node in nodes:
                if (
                    not node.embedding
                    or len(node.embedding) != expected_dim
                    or any(not math.isfinite(v) for v in node.embedding)
                ):
                    raise ValueError("invalid embedding dimension or nonfinite value")
            seen: set[str] = set()
            consumed: set[str] = set()
            patches: dict[str, dict] = {}
            new_by_id = {n.node_id: n for n in nodes}
            for node in nodes:
                if (node.app_id, node.user_id, node.agent_id) != scope:
                    raise ValueError("new node crosses tenant scope")
                if not node.content.strip() or node.embedding is None:
                    raise ValueError("new version requires content and embedding")
                for old_id in node.supersedes:
                    if old_id not in current and old_id not in seen:
                        raise ValueError(
                            "missing, forward or unvalidated version target"
                        )
                    if old_id in consumed:
                        raise ValueError(
                            "two operations consume the same current target"
                        )
                    old = current.get(old_id) or new_by_id[old_id]
                    patch = old.to_metadata()
                    patch.update(
                        is_latest=False,
                        revision=old.revision + 1,
                        superseded_by=_LIST_SEP.join(
                            [*old.superseded_by, node.node_id]
                        ),
                    )
                    patches[old_id] = patch
                    consumed.add(old_id)
                seen.add(node.node_id)
            for nid in archives:
                if nid not in current or nid in consumed:
                    raise ValueError("archive requires one validated unconsumed target")
                old = current[nid]
                patch = old.to_metadata()
                patch.update(
                    is_latest=False,
                    revision=old.revision + 1,
                    custom=json.dumps({**(old.custom or {}), "archived": True}),
                )
                patches[nid] = patch
                consumed.add(nid)
            # A derived claim supported by a retired fact must stop being current.
            # Remaining evidence becomes fresh for the next explicit digest.
            if consumed:
                for derived in self.get_many(
                    {
                        "app_id": scope[0],
                        "user_id": scope[1],
                        "agent_id": scope[2],
                        "layer": Layer.L3_DERIVED.value,
                        "is_latest": True,
                    },
                    limit=100_000,
                ):
                    evidence = set((derived.custom or {}).get("source_ids", []))
                    stale = evidence.intersection(consumed)
                    if stale and derived.node_id not in consumed:
                        patch = derived.to_metadata()
                        patch.update(
                            is_latest=False,
                            revision=derived.revision + 1,
                            custom=json.dumps(
                                {
                                    **(derived.custom or {}),
                                    "stale_evidence": sorted(stale),
                                }
                            ),
                        )
                        patches[derived.node_id] = patch
                        for fid in evidence - consumed:
                            fact = self.get(fid)
                            if fact and fact.is_latest:
                                patches.setdefault(fid, fact.to_metadata())[
                                    "s2_evidence_count"
                                ] = 0
            for fid in processed_evidence or []:
                fact = current.get(fid)
                if (
                    fact is None
                    or not fact.is_latest
                    or fact.layer not in (Layer.L1_FACT, Layer.L2_PROFILE)
                ):
                    raise ValueError(
                        "processed evidence must be a validated current fact"
                    )
                patches.setdefault(fid, fact.to_metadata())["s2_evidence_count"] = 1
            if not nodes and not archives and not processed_evidence:
                return True
            semantic = [
                {
                    "id": n.node_id,
                    "content": n.content,
                    "layer": n.layer.value,
                    "kind": n.kind,
                    "supersedes": n.supersedes,
                    "source_ids": (n.custom or {}).get("source_ids", []),
                    "valid_from": n.valid_from,
                    "valid_until": n.valid_until,
                    "update_type": (n.custom or {}).get("update_type"),
                    "conflicts_with": (n.custom or {}).get("conflicts_with", []),
                    "owner": n.owner,
                    "session_id": n.session_id,
                    "memory_at": n.memory_at,
                    "tags": n.tags,
                    "temporal_scope": (n.custom or {}).get("temporal_scope"),
                    "negation": (n.custom or {}).get("negation", False),
                }
                for n in nodes
            ]
            audit = [
                {
                    "event": "SUPERSEDE" if n.supersedes else "ADD",
                    "node_id": n.node_id,
                    "user_id": n.user_id,
                    "old": None,
                    "new": patches.get(n.node_id, n.to_metadata()),
                }
                for n in nodes
            ]
            audit.extend(
                {
                    "event": "ARCHIVE" if nid in archives else "LINK",
                    "node_id": nid,
                    "user_id": scope[1],
                    "old": current[nid].to_metadata() if nid in current else None,
                    "new": patch,
                }
                for nid, patch in patches.items()
                if nid not in new_ids
            )
            payload = {
                "decisions": decisions or [],
                "request_key": request_key,
                "semantic": semantic,
                "audit": audit,
                "nodes": [
                    {
                        "id": n.node_id,
                        "content": n.content,
                        "embedding": n.embedding,
                        "meta": n.to_metadata(),
                    }
                    for n in nodes
                ],
                "patches": patches,
            }
            with self._journal:
                self._journal.execute(
                    "INSERT INTO plans(id,scope,epoch,payload) VALUES (?,?,?,?)",
                    (plan_id, scope_key, epoch, json.dumps(payload)),
                )
            self._finish_plan(plan_id, scope_key, epoch, payload)
            return True
