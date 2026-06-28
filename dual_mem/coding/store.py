# -*- coding: utf-8 -*-
"""Coding memory store — SQLite (metadata) + VDB (vectors) dual-layer.

Each CodingMemory has 1 SQLite row + 1+N VDB rows (task + search_keys).
Separate VDB collection from chat memories.
"""
import asyncio
import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dual_mem.providers.embedding import EmbedService
from dual_mem.storage.vector_store import VectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus
from dual_mem.coding.types import CodingMemory

logger = logging.getLogger("dual_mem.coding.store")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS coding_memory_meta (
    memory_id        TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    agent_id         TEXT NOT NULL DEFAULT 'default_agent',
    task             TEXT NOT NULL,
    search_keys      TEXT NOT NULL DEFAULT '[]',
    solution         TEXT NOT NULL DEFAULT '',
    boundary_envs    TEXT NOT NULL DEFAULT '',
    boundary_scope   TEXT NOT NULL DEFAULT 'project',
    workspace_id     TEXT,
    branch           TEXT,
    session_id       TEXT,
    files            TEXT NOT NULL DEFAULT '[]',
    confidence       REAL NOT NULL DEFAULT 0.7,
    source           TEXT NOT NULL DEFAULT 'auto_extract',
    created_at       TEXT,
    updated_at       TEXT
);
"""


class CodingMemoryStore:
    """Dual-layer store: SQLite metadata + VDB vectors for task + search_keys."""

    LAYER = Layer.L2_FACT  # reuse L2 for VDB routing; custom flag in node.custom

    def __init__(
        self,
        *,
        db_path: str,
        vector: VectorStore,
        embed: EmbedService,
        collection: str = "coding_memory",
    ):
        self.db_path = db_path
        self.vector = vector
        self.embed = embed
        self.collection = collection
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    async def add(self, mem: CodingMemory) -> str:
        """Insert a new coding memory (SQLite + VDB)."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO coding_memory_meta
                   (memory_id, user_id, agent_id, task, search_keys, solution,
                    boundary_envs, boundary_scope, workspace_id, branch, session_id,
                    files, confidence, source, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mem.memory_id, mem.user_id, mem.agent_id, mem.task,
                    json.dumps(mem.search_keys, ensure_ascii=False),
                    mem.solution, mem.boundary_envs, mem.boundary_scope,
                    mem.workspace_id, mem.branch, mem.session_id,
                    json.dumps(mem.files, ensure_ascii=False),
                    mem.confidence, mem.source,
                    mem.created_at.isoformat() if mem.created_at else None,
                    mem.updated_at.isoformat() if mem.updated_at else None,
                ),
            )
            self._conn.commit()

        await self._upsert_vectors(mem)
        return mem.memory_id

    async def update(self, mem: CodingMemory) -> None:
        """Overwrite an existing memory (SQLite update + VDB re-embed)."""
        old = self.get(mem.memory_id)
        if old:
            await self._delete_vectors(old)
        await self.add(mem)

    async def delete(self, memory_id: str) -> None:
        """Delete a memory (SQLite + VDB)."""
        mem = self.get(memory_id)
        if mem:
            await self._delete_vectors(mem)
        with self._lock:
            self._conn.execute("DELETE FROM coding_memory_meta WHERE memory_id = ?", (memory_id,))
            self._conn.commit()

    def get(self, memory_id: str) -> Optional[CodingMemory]:
        """Fetch one memory by ID from SQLite."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM coding_memory_meta WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_mem(row) if row else None

    def list_by_user(
        self, *, user_id: str, agent_id: str = "default_agent", limit: int = 100
    ) -> List[CodingMemory]:
        """List all coding memories for a user."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM coding_memory_meta
                   WHERE user_id = ? AND agent_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, agent_id, limit),
            ).fetchall()
        return [self._row_to_mem(r) for r in rows]

    async def search(
        self, *, query: str, user_id: str, agent_id: str = "default_agent", top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Semantic search over coding memories via VDB."""
        embedding = await self.embed.embed_batch([query])
        if not embedding:
            return []
        from dual_mem.isolation import build_filter
        where = build_filter(
            app_ids=[self.collection],
            user_id=user_id,
            agent_ids=[agent_id],
            layers=[self.LAYER],
            statuses=[MemoryStatus.ACTIVE],
        )
        nodes = await asyncio.to_thread(
            self.vector.query, embedding=embedding[0], where=where, top_k=top_k,
        )
        results = []
        for node in nodes:
            mem_id = (node.custom or {}).get("coding_memory_id", "")
            if not mem_id:
                continue
            mem = self.get(mem_id)
            if mem:
                results.append({**mem.to_dict(), "score": node.score})
        return results

    async def _upsert_vectors(self, mem: CodingMemory) -> None:
        """Embed task + search_keys and upsert to VDB."""
        texts = [mem.task] + list(mem.search_keys)
        embeddings = await self.embed.embed_batch(texts)
        nodes = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            node = MemoryNode(
                content=text,
                layer=self.LAYER,
                app_id=self.collection,
                user_id=mem.user_id,
                agent_id=mem.agent_id,
                node_id=f"{mem.memory_id}_key{i}",
                tags=["coding"],
                status=MemoryStatus.ACTIVE,
                embedding=emb,
                custom={"coding_memory_id": mem.memory_id, "coding_key_idx": i},
            )
            nodes.append(node)
        await asyncio.to_thread(self.vector.upsert, nodes)

    async def _delete_vectors(self, mem: CodingMemory) -> None:
        """Delete VDB nodes for a memory."""
        n_keys = 1 + len(mem.search_keys)
        node_ids = [f"{mem.memory_id}_key{i}" for i in range(n_keys)]
        await asyncio.to_thread(self.vector.delete, node_ids)

    @staticmethod
    def _row_to_mem(row: sqlite3.Row) -> CodingMemory:
        return CodingMemory(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            task=row["task"],
            search_keys=json.loads(row["search_keys"]),
            solution=row["solution"],
            boundary_envs=row["boundary_envs"],
            boundary_scope=row["boundary_scope"],
            workspace_id=row["workspace_id"],
            branch=row["branch"],
            session_id=row["session_id"],
            files=json.loads(row["files"]),
            confidence=row["confidence"],
            source=row["source"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
