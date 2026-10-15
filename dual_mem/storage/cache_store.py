# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SQLite-backed cache store holding profile caches, System2
task queue, pipeline logs and memory-operation audit records.
"""
import json
import sqlite3
import threading
import time

from dual_mem.storage.sqlite_util import connect_sqlite

_DDL = """
CREATE TABLE IF NOT EXISTS profile_cache (
    iso_key TEXT PRIMARY KEY,
    data    TEXT NOT NULL,
    ts      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS s2_queue (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT NOT NULL,
    app_id    TEXT NOT NULL,
    agent_id  TEXT NOT NULL DEFAULT '',
    status    TEXT NOT NULL DEFAULT 'pending',
    ts        REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pipeline_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    stage      TEXT NOT NULL,
    payload    TEXT NOT NULL,
    ts         REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_operations (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    op      TEXT NOT NULL,
    node_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    ts      REAL NOT NULL
);
"""


class CacheStore:
    """SQLite store for caches, explicit task queue, pipeline logs and audit records."""

    def __init__(self, storage_dir: str):
        self._lock = threading.RLock()
        with self._lock:
            self.conn = connect_sqlite(f"{storage_dir}/cache.db")
            self.conn.row_factory = sqlite3.Row
            self.conn.executescript(_DDL)
            self.conn.commit()

    # ---- Profile cache --------------------------------------------------------------

    def set_profile(self, iso_key: str, data: dict) -> None:
        """Store (or replace) the cached profile for an isolation key."""
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO profile_cache (iso_key, data, ts) VALUES (?, ?, ?)",
                (iso_key, json.dumps(data, ensure_ascii=False), time.time()),
            )
            self.conn.commit()

    def get_profile(self, iso_key: str) -> dict | None:
        """Return the cached profile for an isolation key, or None if missing."""
        with self._lock:
            row = self.conn.execute(
                "SELECT data FROM profile_cache WHERE iso_key = ?", (iso_key,)
            ).fetchone()
            return json.loads(row["data"]) if row else None

    # ---- System2 queue (cognition tasks) --------------------------------------------

    def enqueue_s2_task(
        self,
        user_id: str,
        app_id: str,
        agent_id: str = "",
    ) -> None:
        """Append a pending cognition task, deduplicated per app/user/agent scope."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO s2_queue (user_id, app_id, agent_id, status, ts) "
                "VALUES (?, ?, ?, 'pending', ?)",
                (user_id, app_id, agent_id, time.time()),
            )
            self.conn.commit()

    def list_pending_s2_scopes(self) -> list[dict]:
        """Return pending app/user/agent scopes without changing their status."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, app_id, user_id, agent_id FROM s2_queue "
                "WHERE status = 'pending' ORDER BY id ASC"
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_s2_tasks_done(self, task_ids: list[int]) -> None:
        """Acknowledge only the claimed watermark; newer writes remain queued."""
        if not task_ids:
            return
        with self._lock:
            self.conn.executemany(
                "UPDATE s2_queue SET status='done' WHERE id=? AND status='pending'",
                [(task_id,) for task_id in task_ids],
            )
            self.conn.commit()

    def purge_done_queues(self) -> int:
        """Delete drained System2 rows; return rows removed."""
        with self._lock:
            cur = self.conn.execute("DELETE FROM s2_queue WHERE status = 'done'")
            n_s2 = cur.rowcount
            self.conn.commit()
            return n_s2

    # ---- Pipeline logs --------------------------------------------------------------

    def log_pipeline(self, *, request_id: str, stage: str, payload: dict) -> None:
        """Append a structured pipeline-stage log entry for a request."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO pipeline_logs (request_id, stage, payload, ts) VALUES (?, ?, ?, ?)",
                (
                    request_id,
                    stage,
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )
            self.conn.commit()

    def list_pipeline_logs(self, request_id: str) -> list[dict]:
        """Return all pipeline log entries for a request in order, payloads decoded."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM pipeline_logs WHERE request_id = ? ORDER BY id ASC",
                (request_id,),
            ).fetchall()
            logs = []
            for row in rows:
                item = dict(row)
                item["payload"] = json.loads(item["payload"])
                logs.append(item)
            return logs

    def record_operation(self, *, op: str, node_id: str, user_id: str) -> None:
        """Append a memory-operation audit record."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO memory_operations (op, node_id, user_id, ts) VALUES (?, ?, ?, ?)",
                (op, node_id, user_id, time.time()),
            )
            self.conn.commit()
