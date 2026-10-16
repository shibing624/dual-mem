"""SQLite owns publication/history; vectors are a rebuildable retrieval index.

Candidate rows are reserved but invisible until every vector and link operation
succeeds and a scope CAS transaction publishes the entire batch. Failed vectors
may remain indexed, but coding reads always gate them through published SQLite
rows. History stays ACTIVE in the VDB; coding search uses SQL is_latest, never
vector hints, so partial link metadata cannot retire a visible head.
"""

import asyncio
import json
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from dual_mem.coding.types import (
    BOUNDARY_SCOPES,
    CodingConcurrentWriteError,
    CodingDecisionError,
    CodingMemory,
    CodingScope,
    coding_scope,
    version_boundary,
)
from dual_mem.errors import IdempotencyConflictError
from dual_mem.isolation import build_filter
from dual_mem.providers.embedding import EmbedService
from dual_mem.storage.vector_store import VectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS coding_memory_meta (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT 'default_agent',
    app_id TEXT NOT NULL DEFAULT 'default',
    task TEXT NOT NULL,
    search_keys TEXT NOT NULL DEFAULT '[]',
    solution TEXT NOT NULL DEFAULT '',
    boundary_envs TEXT NOT NULL DEFAULT '',
    boundary_scope TEXT NOT NULL DEFAULT 'project',
    workspace_id TEXT,
    branch TEXT,
    session_id TEXT,
    files TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.7,
    source TEXT NOT NULL DEFAULT 'auto_extract',
    created_at TEXT,
    updated_at TEXT,
    is_latest INTEGER NOT NULL DEFAULT 1,
    supersedes TEXT NOT NULL DEFAULT '[]',
    superseded_by TEXT NOT NULL DEFAULT '[]',
    source_ids TEXT NOT NULL DEFAULT '[]',
    published INTEGER NOT NULL DEFAULT 1
)
"""


class CodingMemoryStore:
    LAYER = Layer.L1_FACT
    SCAN_LIMIT = 100_000

    def __init__(
        self, *, db_path: str, vector: VectorStore, embed: EmbedService
    ) -> None:
        self.db_path = db_path
        self.vector = vector
        self.embed = embed
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._init_db()
        except BaseException:
            self._conn.close()
            raise

    def _init_db(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(_CREATE_TABLE_SQL)
            columns = {
                r[1]
                for r in self._conn.execute("PRAGMA table_info(coding_memory_meta)")
            }
            required = {
                "is_latest",
                "supersedes",
                "superseded_by",
                "source_ids",
                "published",
            }
            if not required <= columns:
                raise CodingDecisionError(
                    "unsupported coding schema; use a new storage directory"
                )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_coding_user ON coding_memory_meta (app_id,user_id,agent_id)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS coding_epochs (scope TEXT PRIMARY KEY, epoch INTEGER NOT NULL)"
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS coding_sources (
                raw_id TEXT PRIMARY KEY, app_id TEXT NOT NULL, user_id TEXT NOT NULL,
                agent_id TEXT NOT NULL, messages TEXT NOT NULL, session_id TEXT,
                workspace_id TEXT, branch TEXT, indexed INTEGER NOT NULL DEFAULT 0
            )"""
            )
            # Written in the same transaction that publishes a request's batch,
            # so a committed request can always be replayed instead of re-decided.
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS coding_requests (request_key TEXT PRIMARY KEY, result TEXT NOT NULL, ts REAL NOT NULL)"
            )
            self._validate_history()

    def _validate_history(self) -> None:
        """Validate persisted history without rewriting links or visibility."""
        rows = self._conn.execute(
            "SELECT * FROM coding_memory_meta WHERE published=1"
        ).fetchall()
        memories = {r["memory_id"]: self._row_to_mem(r) for r in rows}
        reverse: dict[str, list[str]] = {mid: [] for mid in memories}
        for mem in memories.values():
            for old_id in mem.supersedes:
                old = memories.get(old_id)
                if old is None or coding_scope(old) != coding_scope(mem):
                    raise CodingDecisionError("missing or foreign coding history")
                if version_boundary(old) != version_boundary(mem):
                    raise CodingDecisionError(
                        "coding history crosses retirement boundary"
                    )
                reverse[old_id].append(mem.memory_id)
        # Kahn traversal detects cycles without recursion or truncating history.
        counts = {mid: len(set(m.supersedes)) for mid, m in memories.items()}
        ready = [mid for mid, count in counts.items() if count == 0]
        visited = 0
        while ready:
            mid = ready.pop()
            visited += 1
            for child in set(reverse[mid]):
                counts[child] -= 1
                if counts[child] == 0:
                    ready.append(child)
        if visited != len(memories):
            raise CodingDecisionError("cyclic coding history")
        for mid, mem in memories.items():
            links = list(dict.fromkeys(reverse[mid]))
            if set(mem.superseded_by) != set(links) or (links and mem.is_latest):
                raise CodingDecisionError(
                    "inconsistent coding reverse history or visibility"
                )

    @staticmethod
    def _vdb_app_id(app_id: str) -> str:
        return f"coding_{app_id}"

    @staticmethod
    def _scope_key(scope: CodingScope) -> str:
        return json.dumps(scope, ensure_ascii=False)

    def _epoch(self, scope: CodingScope) -> int:
        row = self._conn.execute(
            "SELECT epoch FROM coding_epochs WHERE scope=?", (self._scope_key(scope),)
        ).fetchone()
        return row[0] if row else 0

    def assert_epoch(self, scope: CodingScope, expected_epoch: int) -> None:
        with self._lock:
            if self._epoch(scope) != expected_epoch:
                raise CodingConcurrentWriteError(
                    "coding scope changed after reconciliation"
                )

    async def retain_source(
        self,
        messages: list[dict],
        *,
        user_id: str,
        agent_id: str = "default_agent",
        app_id: str = "default",
        session_id: str | None = None,
        workspace_id: str | None = None,
        branch: str | None = None,
        raw_id: str | None = None,
    ) -> str:
        """Persist lossless source BEFORE generation; never discard it on failure.

        With an explicit ``raw_id`` (idempotent requests) a retry reuses the stored
        source when it matches and rejects different content under the same id.
        """
        raw_id = raw_id or str(uuid.uuid4())
        content = json.dumps(messages, ensure_ascii=False)
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT app_id,user_id,agent_id,messages,indexed FROM coding_sources WHERE raw_id=?",
                (raw_id,),
            ).fetchone()
            if existing is not None:
                if tuple(existing[:4]) != (app_id, user_id, agent_id, content):
                    raise IdempotencyConflictError(
                        "coding source ID reused with different input"
                    )
                if existing["indexed"]:
                    return raw_id
            else:
                self._conn.execute(
                    "INSERT INTO coding_sources (raw_id,app_id,user_id,agent_id,messages,session_id,workspace_id,branch) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        raw_id,
                        app_id,
                        user_id,
                        agent_id,
                        content,
                        session_id,
                        workspace_id,
                        branch,
                    ),
                )
        embeddings = await self.embed.embed_batch([content])
        self._validate_embeddings(embeddings, 1)
        node = MemoryNode(
            node_id=raw_id,
            content=content,
            layer=Layer.L0_RAW,
            app_id=self._vdb_app_id(app_id),
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id or "",
            embedding=list(embeddings[0]),
            tags=["coding"],
            custom={"coding_raw_id": raw_id},
        )
        if not self.vector.get_by_ids([raw_id]):
            await asyncio.to_thread(self._write_vectors, [node])
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE coding_sources SET indexed=1 WHERE raw_id=?", (raw_id,)
            )
        return raw_id

    def request_result(self, request_key: str) -> dict[str, Any] | None:
        """Result recorded atomically with a published request batch."""
        with self._lock:
            row = self._conn.execute(
                "SELECT result FROM coding_requests WHERE request_key=?", (request_key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def purge_requests(self, *, older_than: float) -> int:
        """Drop replay records older than ``older_than`` (unix seconds)."""
        with self._lock, self._conn:
            return self._conn.execute(
                "DELETE FROM coding_requests WHERE ts < ?", (older_than,)
            ).rowcount

    def discard_unpublished(self, memory_ids: list[str]) -> None:
        """Drop reserved-but-unpublished rows and vectors so a retry can re-reserve.

        Unpublished rows are never visible to reads; only an interrupted attempt
        (crash between reservation and publish) leaves them behind.
        """
        if not memory_ids:
            return
        marks = ",".join("?" for _ in memory_ids)
        with self._lock, self._conn:
            rows = self._conn.execute(
                f"SELECT memory_id, search_keys FROM coding_memory_meta "
                f"WHERE published=0 AND memory_id IN ({marks})",
                memory_ids,
            ).fetchall()
            if not rows:
                return
            self._conn.execute(
                f"DELETE FROM coding_memory_meta WHERE published=0 AND memory_id IN ({marks})",
                memory_ids,
            )
        keys = [
            f"{row['memory_id']}_key{i}"
            for row in rows
            for i in range(1 + len(json.loads(row["search_keys"])))
        ]
        self.vector.delete(keys)

    def record_request(self, request_key: str, result: dict[str, Any]) -> None:
        """Record a request whose decisions changed nothing (no batch to publish)."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO coding_requests (request_key,result,ts) VALUES (?,?,?)",
                (request_key, json.dumps(result, ensure_ascii=False), time.time()),
            )

    def get_source(
        self,
        raw_id: str,
        *,
        user_id: str,
        agent_id: str = "default_agent",
        app_id: str = "default",
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM coding_sources WHERE raw_id=? AND app_id=? AND user_id=? AND agent_id=?",
                (raw_id, app_id, user_id, agent_id),
            ).fetchone()
        return dict(row, messages=json.loads(row["messages"])) if row else None

    def get(self, memory_id: str) -> CodingMemory | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM coding_memory_meta WHERE memory_id=? AND published=1",
                (memory_id,),
            ).fetchone()
        return self._row_to_mem(row) if row else None

    def list_by_user(
        self,
        *,
        user_id: str,
        agent_id: str = "default_agent",
        app_id: str = "default",
        limit: int | None = None,
    ) -> list[CodingMemory]:
        _, current, _ = self.snapshot(user_id=user_id, agent_id=agent_id, app_id=app_id)
        if limit is not None and limit < 0:
            raise ValueError("coding limit must be nonnegative")
        return current if limit is None else current[:limit]

    def snapshot(
        self, *, user_id: str, agent_id: str = "default_agent", app_id: str = "default"
    ) -> tuple[int, list[CodingMemory], dict[str, list[dict[str, Any]]]]:
        """One SQLite read transaction: complete scope, heads, ancestors and epoch."""
        scope = (app_id, user_id, agent_id)
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                epoch = self._epoch(scope)
                rows = self._conn.execute(
                    "SELECT * FROM coding_memory_meta WHERE app_id=? AND user_id=? AND agent_id=? AND published=1 ORDER BY created_at DESC,memory_id LIMIT ?",
                    (*scope, self.SCAN_LIMIT + 1),
                ).fetchall()
                if len(rows) > self.SCAN_LIMIT:
                    raise CodingDecisionError(
                        "coding scope exceeds complete scan budget"
                    )
                memories = {r["memory_id"]: self._row_to_mem(r) for r in rows}
                heads = [m for m in memories.values() if m.is_latest]
                histories = {m.memory_id: self._history(m, memories) for m in heads}
                return epoch, heads, histories
            finally:
                self._conn.rollback()

    @staticmethod
    def _history(
        head: CodingMemory, memories: dict[str, CodingMemory]
    ) -> list[dict[str, Any]]:
        found: dict[str, CodingMemory] = {}
        visiting: set[str] = set()

        def visit(memory: CodingMemory) -> None:
            if memory.memory_id in visiting:
                raise CodingDecisionError("cyclic coding history")
            visiting.add(memory.memory_id)
            for old_id in memory.supersedes:
                old = memories.get(old_id)
                if old is None:
                    raise CodingDecisionError("missing or foreign coding history")
                if old.is_latest or version_boundary(old) != version_boundary(memory):
                    raise CodingDecisionError(
                        "coding history crosses retirement boundary"
                    )
                if memory.memory_id not in old.superseded_by:
                    raise CodingDecisionError("missing coding reverse history link")
                if old_id not in found:
                    visit(old)
                    found[old_id] = old
            visiting.remove(memory.memory_id)

        visit(head)
        return [m.to_dict() for m in found.values()]

    async def add(self, mem: CodingMemory) -> str:
        await self.add_many([mem])
        return mem.memory_id

    async def add_many(
        self,
        memories: list[CodingMemory],
        *,
        expected_epoch: int | None = None,
        archive_ids: list[str] | None = None,
        request: tuple[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        """Append an all-or-nothing visible batch; immutable IDs cannot be reused.

        ``request`` = (request_key, result) is recorded in the publishing transaction.
        """
        archive_ids = list(archive_ids or [])
        if not memories and not archive_ids:
            if request:
                self.record_request(*request)
            return []
        if not memories:
            head = self.get(archive_ids[0])
            if head is None:
                raise ValueError("coding archive target missing")
            scope = coding_scope(head)
            epoch = self._epoch(scope) if expected_epoch is None else expected_epoch
            await asyncio.to_thread(
                self._publish, [], scope, epoch, archive_ids, request
            )
            return []
        # Freeze the caller's mutable dataclasses before any await.
        memories = [
            replace(
                m,
                search_keys=list(m.search_keys),
                files=list(m.files),
                supersedes=list(m.supersedes),
                superseded_by=list(m.superseded_by),
                source_ids=list(m.source_ids),
            )
            for m in memories
        ]
        scope = coding_scope(memories[0])
        with self._lock:
            epoch = self._epoch(scope) if expected_epoch is None else expected_epoch
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._validate_plan(memories, scope, epoch)
                for mem in memories:
                    self._insert(mem, published=False)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        texts = [text for mem in memories for text in [mem.task, *mem.search_keys]]
        embeddings = await self.embed.embed_batch(texts)
        self._validate_embeddings(embeddings, len(texts))
        nodes = []
        offset = 0
        for mem in memories:
            for i, text in enumerate([mem.task, *mem.search_keys]):
                nodes.append(
                    MemoryNode(
                        node_id=f"{mem.memory_id}_key{i}",
                        content=text,
                        layer=self.LAYER,
                        app_id=self._vdb_app_id(mem.app_id),
                        user_id=mem.user_id,
                        agent_id=mem.agent_id,
                        embedding=list(embeddings[offset]),
                        tags=["coding"],
                        supersedes=[f"{old}_key0" for old in mem.supersedes],
                        custom={
                            "coding_memory_id": mem.memory_id,
                            "coding_key_idx": i,
                            "source_ids": mem.source_ids,
                        },
                    )
                )
                offset += 1
        await asyncio.to_thread(self._write_vectors, nodes)
        await asyncio.to_thread(
            self._publish, memories, scope, epoch, archive_ids, request
        )
        return [m.memory_id for m in memories]

    def _validate_plan(
        self,
        memories: list[CodingMemory],
        scope: CodingScope,
        epoch: int,
        *,
        reserved: bool = False,
    ) -> None:
        if self._epoch(scope) != epoch:
            raise CodingConcurrentWriteError(
                "coding scope changed after reconciliation"
            )
        ids = [m.memory_id for m in memories]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate coding memory IDs")
        retired = set()
        for mem in memories:
            if coding_scope(mem) != scope or mem.boundary_scope not in BOUNDARY_SCOPES:
                raise ValueError("coding memory crosses scope")
            if not isinstance(mem.task, str) or not isinstance(mem.solution, str):
                raise ValueError("coding text must be strings")
            if any(
                not isinstance(key, str) or not key.strip() for key in mem.search_keys
            ):
                raise ValueError("invalid coding search keys")
            if not math.isfinite(mem.confidence) or not 0 <= mem.confidence <= 1:
                raise ValueError("invalid coding confidence")
            if (
                not mem.memory_id
                or not mem.task.strip()
                or not mem.solution.strip()
                or not mem.is_latest
                or mem.superseded_by
            ):
                raise ValueError("coding append requires a complete new head")
            row = self._conn.execute(
                "SELECT published FROM coding_memory_meta WHERE memory_id=?",
                (mem.memory_id,),
            ).fetchone()
            if (reserved and (row is None or row[0])) or (
                not reserved and row is not None
            ):
                raise ValueError("immutable coding memory ID already used")
            if len(set(mem.supersedes)) != len(mem.supersedes):
                raise ValueError("duplicate coding version targets")
            for source_id in mem.source_ids:
                raw = self._conn.execute(
                    "SELECT app_id,user_id,agent_id,indexed FROM coding_sources WHERE raw_id=?",
                    (source_id,),
                ).fetchone()
                if raw is None or tuple(raw[:3]) != scope or not raw[3]:
                    raise ValueError(
                        "coding source is missing, unindexed or crosses scope"
                    )
            for old_id in mem.supersedes:
                old = self.get(old_id)
                if old is None or coding_scope(old) != scope:
                    raise ValueError("coding version target missing or crosses scope")
                if not old.is_latest or old_id in retired:
                    raise CodingConcurrentWriteError(
                        "coding version target is no longer current"
                    )
                if version_boundary(mem) != version_boundary(old):
                    raise ValueError("coding retirement crosses version boundary")
                retired.add(old_id)

    def _publish(
        self,
        memories: list[CodingMemory],
        scope: CodingScope,
        epoch: int,
        archive_ids: list[str] | None = None,
        request: tuple[str, dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if memories:
                    self._validate_plan(memories, scope, epoch, reserved=True)
                elif self._epoch(scope) != epoch:
                    raise CodingConcurrentWriteError(
                        "coding scope changed after reconciliation"
                    )
                patches = {}
                retired = {old_id for mem in memories for old_id in mem.supersedes}
                for old_id in archive_ids or []:
                    if old_id in retired:
                        raise CodingConcurrentWriteError(
                            "coding archive target is also replaced in this batch"
                        )
                    old = self.get(old_id)
                    if (
                        old is None
                        or not old.is_latest
                        or coding_scope(old) != scope
                    ):
                        raise ValueError(
                            "coding archive target missing or not current"
                        )
                    retired.add(old_id)
                    for i in range(1 + len(old.search_keys)):
                        nid = f"{old_id}_key{i}"
                        vector = self.vector.get(nid)
                        if vector is None or (
                            vector.app_id,
                            vector.user_id,
                            vector.agent_id,
                        ) != (self._vdb_app_id(scope[0]), scope[1], scope[2]):
                            raise CodingDecisionError(
                                "missing or foreign coding history vector"
                            )
                        patches[nid] = {"is_latest": False, "superseded_by": ""}
                    self._conn.execute(
                        "UPDATE coding_memory_meta SET is_latest=0 WHERE memory_id=? AND published=1 AND is_latest=1",
                        (old_id,),
                    )
                for mem in memories:
                    for old_id in mem.supersedes:
                        old = self.get(old_id)
                        links = [*old.superseded_by, mem.memory_id]
                        for i in range(1 + len(old.search_keys)):
                            nid = f"{old_id}_key{i}"
                            vector = self.vector.get(nid)
                            if vector is None or (
                                vector.app_id,
                                vector.user_id,
                                vector.agent_id,
                            ) != (self._vdb_app_id(scope[0]), scope[1], scope[2]):
                                raise CodingDecisionError(
                                    "missing or foreign coding history vector"
                                )
                            patches[nid] = {
                                "is_latest": False,
                                "superseded_by": "\x1f".join(
                                    f"{link}_key0" for link in links
                                ),
                            }
                        self._conn.execute(
                            "UPDATE coding_memory_meta SET is_latest=0,superseded_by=? WHERE memory_id=? AND published=1 AND is_latest=1",
                            (json.dumps(links), old_id),
                        )
                if patches:
                    self.vector.update_payload_many(patches)
                    checked = self.vector.get_by_ids(list(patches))
                    if set(checked) != set(patches) or any(
                        checked[nid].is_latest
                        or (
                            "superseded_by" in patch
                            and checked[nid].superseded_by
                            != (
                                patch["superseded_by"].split("\x1f")
                                if patch["superseded_by"]
                                else []
                            )
                        )
                        for nid, patch in patches.items()
                    ):
                        raise CodingDecisionError(
                            "coding vector history link acknowledgement missing"
                        )
                for mem in memories:
                    self._conn.execute(
                        "UPDATE coding_memory_meta SET published=1 WHERE memory_id=? AND published=0",
                        (mem.memory_id,),
                    )
                self._conn.execute(
                    "INSERT INTO coding_epochs (scope,epoch) VALUES (?,?) ON CONFLICT(scope) DO UPDATE SET epoch=excluded.epoch",
                    (self._scope_key(scope), epoch + 1),
                )
                if request:
                    self._conn.execute(
                        "INSERT INTO coding_requests (request_key,result,ts) VALUES (?,?,?)",
                        (
                            request[0],
                            json.dumps(request[1], ensure_ascii=False),
                            time.time(),
                        ),
                    )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def _insert(self, mem: CodingMemory, *, published: bool) -> None:
        data = mem.to_dict()
        for key in (
            "search_keys",
            "files",
            "supersedes",
            "superseded_by",
            "source_ids",
        ):
            data[key] = json.dumps(data[key], ensure_ascii=False)
        data["is_latest"] = int(mem.is_latest)
        data["published"] = int(published)
        columns = list(data)
        self._conn.execute(
            f"INSERT INTO coding_memory_meta ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            tuple(data.values()),
        )

    @staticmethod
    def _validate_embeddings(embeddings: list[list[float]], expected: int) -> None:
        if len(embeddings) != expected:
            raise CodingDecisionError("coding embedding count mismatch")
        dim = len(embeddings[0]) if embeddings else 0
        if not dim or any(
            len(e) != dim or any(not math.isfinite(v) for v in e) for e in embeddings
        ):
            raise CodingDecisionError("invalid coding embeddings")

    def _write_vectors(self, nodes: list[MemoryNode]) -> None:
        ids = [n.node_id for n in nodes]
        if self.vector.get_by_ids(ids):
            raise ValueError("immutable coding vector ID already used")
        self.vector.upsert(nodes)
        saved = self.vector.get_by_ids(ids)
        if set(saved) != set(ids) or any(
            saved[n.node_id].content != n.content
            or (
                saved[n.node_id].app_id,
                saved[n.node_id].user_id,
                saved[n.node_id].agent_id,
            )
            != (n.app_id, n.user_id, n.agent_id)
            or saved[n.node_id].supersedes != n.supersedes
            for n in nodes
        ):
            raise CodingDecisionError("coding vector append acknowledgement missing")

    async def update(self, mem: CodingMemory) -> None:
        raise ValueError("coding history is immutable; append a new ID with supersedes")

    async def delete(self, memory_id: str) -> None:
        raise ValueError(
            "coding history is immutable; destructive delete is unsupported"
        )

    async def search(
        self,
        *,
        query: str,
        user_id: str,
        agent_id: str = "default_agent",
        app_id: str = "default",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("coding top_k must be positive")
        embeddings = await self.embed.embed_batch([query])
        self._validate_embeddings(embeddings, 1)
        # Fetch all index keys in this bounded scope, including invisible attempts
        # and history, so neither duplicates nor retired hits consume top_k.
        with self._lock:
            rows = self._conn.execute(
                "SELECT search_keys FROM coding_memory_meta WHERE app_id=? AND user_id=? AND agent_id=? LIMIT ?",
                (app_id, user_id, agent_id, self.SCAN_LIMIT + 1),
            ).fetchall()
        count = sum(1 + len(json.loads(r[0])) for r in rows)
        if count > self.SCAN_LIMIT:
            raise CodingDecisionError("coding search exceeds complete index budget")
        if not count:
            return []
        nodes = await asyncio.to_thread(
            self.vector.query,
            embedding=embeddings[0],
            where=build_filter(
                app_ids=[self._vdb_app_id(app_id)],
                user_id=user_id,
                agent_ids=[agent_id],
                layers=[self.LAYER],
                statuses=[MemoryStatus.ACTIVE],
            ),
            top_k=count,
        )
        results = []
        seen = set()
        for node in nodes:
            mem_id = (node.custom or {}).get("coding_memory_id")
            if mem_id in seen:
                continue
            mem = self.get(mem_id) if mem_id else None
            if (
                mem
                and mem.is_latest
                and coding_scope(mem) == (app_id, user_id, agent_id)
            ):
                seen.add(mem_id)
                results.append(dict(mem.to_dict(), score=node.score))
                if len(results) == top_k:
                    break
        return results

    @staticmethod
    def _row_to_mem(row: sqlite3.Row) -> CodingMemory:
        data = dict(row)
        data.pop("published", None)
        for key in (
            "search_keys",
            "files",
            "supersedes",
            "superseded_by",
            "source_ids",
        ):
            data[key] = json.loads(data[key])
        for key in ("created_at", "updated_at"):
            data[key] = datetime.fromisoformat(data[key]) if data[key] else None
        data["is_latest"] = bool(data["is_latest"])
        return CodingMemory(**data)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
