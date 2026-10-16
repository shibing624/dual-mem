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


def _plan_semantic(nodes: list[MemoryNode]) -> list[dict]:
    """Replay identity of a version plan: everything except embeddings/revisions."""
    return [
        {
            "id": n.node_id,
            "content": n.content,
            "layer": n.layer.value,
            "kind": n.kind,
            "supersedes": n.supersedes,
            "superseded_by": n.superseded_by,
            "is_latest": n.is_latest,
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
        indexed = self._journal.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='requests'"
        ).fetchone()
        # request_key -> plan, written with the plan row; looked up on every write.
        self._journal.execute(
            "CREATE TABLE IF NOT EXISTS requests (request_key TEXT PRIMARY KEY, plan_id TEXT NOT NULL)"
        )
        if not indexed:
            for plan_id, payload in self._journal.execute(
                "SELECT id, payload FROM plans"
            ).fetchall():
                key = json.loads(payload).get("request_key")
                if key:
                    self._journal.execute(
                        "INSERT OR IGNORE INTO requests (request_key, plan_id) VALUES (?, ?)",
                        (key, plan_id),
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

        try:
            with self._guard(recover=False):
                self._validate_schema()
                self._recover_pending()
        except BaseException:
            self._journal.close()
            raise

    @contextmanager
    def _guard(self, *, recover: bool = True) -> Iterator[None]:
        """Serialize cooperating processes and recover before exposing any read."""
        with self._lock:
            if getattr(self._local, "inside", False):
                yield
                return
            with self._lock_path.open("a") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                self._local.inside = True
                try:
                    if recover:
                        self._recover_pending()
                    yield
                finally:
                    self._local.inside = False
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _check_metadata_schema(meta: dict) -> None:
        if (
            meta.get("layer_schema_version") != 4
            or meta.get("layer") not in Layer._value2member_map_
            or meta.get("status") != MemoryStatus.ACTIVE.value
        ):
            raise ValueError("unsupported memory schema; use a new storage directory")

    def _recover_pending(self) -> None:
        """Validate every pending payload before recovering any of them."""
        pending = [
            (plan_id, scope, epoch, json.loads(payload))
            for plan_id, scope, epoch, payload in self._journal.execute(
                "SELECT id, scope, epoch, payload FROM plans WHERE done=0 ORDER BY rowid"
            ).fetchall()
        ]
        for _, _, _, payload in pending:
            for row in payload["nodes"]:
                self._check_metadata_schema(row["meta"])
            for meta in payload["patches"].values():
                self._check_metadata_schema(meta)
        for plan_id, scope, epoch, payload in pending:
            self._finish_plan(plan_id, scope, epoch, payload)

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

    def _validate_schema(self) -> None:
        """Reject unsupported stored metadata without mutating historical rows."""
        offset = 0
        while True:
            rows = self.collection.get(limit=500, offset=offset, include=["metadatas"])
            if not rows["ids"]:
                return
            for meta in rows["metadatas"]:
                self._check_metadata_schema(meta)
            offset += len(rows["ids"])

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
        """The committed result of ``request_key``, or None while it has not committed."""
        with self._guard():
            row = self._journal.execute(
                "SELECT plans.payload FROM requests JOIN plans ON plans.id = requests.plan_id "
                "WHERE requests.request_key = ? AND plans.done = 1",
                (request_key,),
            ).fetchone()
        if row is None:
            return None
        plan = json.loads(row[0])
        return {
            "extra_node_ids": [n["id"] for n in plan["nodes"]],
            "commit_passed": bool(plan["nodes"] or plan["patches"]),
            "decisions": plan.get("decisions", []),
            "needs_review": plan.get("needs_review", []),
        }

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
        needs_review: list[str] | None = None,
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
                if previous.get("semantic") != _plan_semantic(nodes):
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
            linked: set[str] = set()
            patches: dict[str, dict] = {}
            new_by_id = {n.node_id: n for n in nodes}

            def patch_of(nid: str) -> dict:
                if nid not in patches:
                    old = current.get(nid) or new_by_id[nid]
                    patches[nid] = old.to_metadata()
                    patches[nid]["revision"] = old.revision + 1
                return patches[nid]

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
                    if old_id in consumed or old_id in linked:
                        raise ValueError(
                            "two operations consume the same current target"
                        )
                    old = current.get(old_id) or new_by_id[old_id]
                    patch = patch_of(old_id)
                    patch.update(
                        is_latest=False,
                        superseded_by=_LIST_SEP.join(
                            [*old.superseded_by, node.node_id]
                        ),
                    )
                    consumed.add(old_id)
                # An older report is inserted behind a newer head that stays current.
                for newer_id in node.superseded_by:
                    if node.is_latest or newer_id not in current:
                        raise ValueError(
                            "historical version requires a validated newer head"
                        )
                    if newer_id in consumed:
                        raise ValueError(
                            "historical version cannot attach to a retired target"
                        )
                    newer = current[newer_id]
                    patch = patch_of(newer_id)
                    existing = patch["supersedes"].split(_LIST_SEP) if patch["supersedes"] else []
                    patch["supersedes"] = _LIST_SEP.join([*existing, node.node_id])
                    linked.add(newer_id)
                seen.add(node.node_id)
            for nid in archives:
                if nid not in current or nid in consumed or nid in linked:
                    raise ValueError("archive requires one validated unconsumed target")
                old = current[nid]
                patch = patch_of(nid)
                patch.update(
                    is_latest=False,
                    custom=json.dumps({**(old.custom or {}), "archived": True}),
                )
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
                if request_key:
                    # Nothing changes, but the request has completed: record it so a
                    # retry replays it and deferred reviews know it has finished.
                    empty_id = hashlib.sha256(
                        f"request:{request_key}".encode()
                    ).hexdigest()
                    with self._journal:
                        self._journal.execute(
                            "INSERT OR IGNORE INTO plans(id,scope,epoch,payload,done) VALUES (?,?,?,?,1)",
                            (
                                empty_id,
                                scope_key,
                                epoch,
                                json.dumps(
                                    {
                                        "decisions": decisions or [],
                                        "needs_review": needs_review or [],
                                        "request_key": request_key,
                                        "nodes": [],
                                        "patches": {},
                                    }
                                ),
                            ),
                        )
                        self._journal.execute(
                            "INSERT OR IGNORE INTO requests (request_key, plan_id) VALUES (?, ?)",
                            (request_key, empty_id),
                        )
                return True
            semantic = _plan_semantic(nodes)
            audit = [
                {
                    "event": (
                        "HISTORICAL"
                        if n.superseded_by
                        else ("SUPERSEDE" if n.supersedes else "ADD")
                    ),
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
                "needs_review": needs_review or [],
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
                if request_key:
                    self._journal.execute(
                        "INSERT OR IGNORE INTO requests (request_key, plan_id) VALUES (?, ?)",
                        (request_key, plan_id),
                    )
            self._finish_plan(plan_id, scope_key, epoch, payload)
            return True
