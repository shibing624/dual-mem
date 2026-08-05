# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SQLite-backed cache store holding profile caches, System2/intention/reconcile
task queues, pipeline logs and memory-operation audit records.
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
CREATE TABLE IF NOT EXISTS reconcile_queue (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  TEXT NOT NULL,
    app_id   TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT '',
    node_ids TEXT NOT NULL,
    status   TEXT NOT NULL DEFAULT 'pending',
    ts       REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS intention_queue (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    app_id  TEXT NOT NULL,
    status  TEXT NOT NULL DEFAULT 'pending',
    ts      REAL NOT NULL
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
CREATE TABLE IF NOT EXISTS content_hash_cache (
    iso_key      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    ts           REAL NOT NULL,
    PRIMARY KEY (iso_key, content_hash)
);
"""


class CacheStore:
    """SQLite store for caches, explicit task queues, pipeline logs and audit records."""

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
            existing = self.conn.execute(
                "SELECT 1 FROM s2_queue WHERE status = 'pending' "
                "AND user_id = ? AND app_id = ? AND agent_id = ? LIMIT 1",
                (user_id, app_id, agent_id),
            ).fetchone()
            if existing is not None:
                return
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
                "SELECT app_id, user_id, agent_id FROM s2_queue "
                "WHERE status = 'pending' ORDER BY id ASC"
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_s2_scope_done(self, *, app_id: str, user_id: str, agent_id: str) -> None:
        """Acknowledge a cognition scope after its explicit digest succeeds."""
        with self._lock:
            self.conn.execute(
                "UPDATE s2_queue SET status = 'done' WHERE status = 'pending' "
                "AND app_id = ? AND user_id = ? AND agent_id = ?",
                (app_id, user_id, agent_id),
            )
            self.conn.commit()

    # ---- Reconcile queue (asynchronous evolution chain merging) ---------------------

    def enqueue_reconcile_task(
        self, *, app_id: str, user_id: str, agent_id: str, node_ids: list[str]
    ) -> None:
        """Append a reconcile task carrying the freshly written node ids that need merging."""
        if not node_ids:
            return
        with self._lock:
            self.conn.execute(
                "INSERT INTO reconcile_queue (app_id, user_id, agent_id, node_ids, status, ts) "
                "VALUES (?, ?, ?, ?, 'pending', ?)",
                (app_id, user_id, agent_id, json.dumps(node_ids, ensure_ascii=False), time.time()),
            )
            self.conn.commit()

    def list_pending_reconcile_scopes(self) -> list[dict]:
        """Distinct (app_id, user_id, agent_id) with pending reconcile tasks."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT DISTINCT app_id, user_id, agent_id "
                "FROM reconcile_queue WHERE status = 'pending'"
            ).fetchall()
            return [dict(row) for row in rows]

    def list_pending_reconcile_tasks(
        self, *, app_id: str | None = None, user_id: str | None = None, agent_id: str | None = None
    ) -> list[dict]:
        """Return pending reconcile tasks without acknowledging them."""
        with self._lock:
            clauses = ["status = 'pending'"]
            params: list = []
            if app_id is not None:
                clauses.append("app_id = ?")
                params.append(app_id)
            if user_id is not None:
                clauses.append("user_id = ?")
                params.append(user_id)
            if agent_id is not None:
                clauses.append("agent_id = ?")
                params.append(agent_id)
            sql = (
                f"SELECT * FROM reconcile_queue WHERE {' AND '.join(clauses)} "
                "ORDER BY id ASC"
            )
            rows = self.conn.execute(sql, params).fetchall()
            tasks = [dict(row) for row in rows]
            for task in tasks:
                task["node_ids"] = json.loads(task["node_ids"]) if task["node_ids"] else []
            return tasks

    def mark_reconcile_task_done(self, task_id: int) -> None:
        """Acknowledge one reconcile task after it has completed successfully."""
        with self._lock:
            self.conn.execute(
                "UPDATE reconcile_queue SET status = 'done' "
                "WHERE id = ? AND status = 'pending'",
                (task_id,),
            )
            self.conn.commit()

    def reconcile_queue_size(self) -> int:
        """Return how many reconcile tasks are still pending across all users."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM reconcile_queue WHERE status = 'pending'"
            ).fetchone()
            return int(row["n"]) if row else 0

    def purge_done_queues(self) -> int:
        """Delete drained reconcile/s2 rows; return rows removed."""
        with self._lock:
            cur = self.conn.execute("DELETE FROM reconcile_queue WHERE status = 'done'")
            n_reconcile = cur.rowcount
            cur = self.conn.execute("DELETE FROM s2_queue WHERE status = 'done'")
            n_s2 = cur.rowcount
            self.conn.commit()
            return n_reconcile + n_s2

    # ---- Pipeline logs --------------------------------------------------------------

    def log_pipeline(self, *, request_id: str, stage: str, payload: dict) -> None:
        """Append a structured pipeline-stage log entry for a request."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO pipeline_logs (request_id, stage, payload, ts) VALUES (?, ?, ?, ?)",
                (request_id, stage, json.dumps(payload, ensure_ascii=False), time.time()),
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

    # ---- Content-hash write dedup ---------------------------------------------------

    def get_content_hash_outcome(self, iso_key: str, content_hash: str) -> dict | None:
        """Return cached WriterOutcome fields for (scope, content_hash), or None."""
        with self._lock:
            row = self.conn.execute(
                "SELECT outcome_json FROM content_hash_cache WHERE iso_key = ? AND content_hash = ?",
                (iso_key, content_hash),
            ).fetchone()
            if not row:
                return None
            return json.loads(row["outcome_json"])

    def set_content_hash_outcome(
        self,
        iso_key: str,
        content_hash: str,
        outcome: dict,
    ) -> None:
        """Persist write outcome for duplicate-content short-circuit."""
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO content_hash_cache (iso_key, content_hash, outcome_json, ts) "
                "VALUES (?, ?, ?, ?)",
                (iso_key, content_hash, json.dumps(outcome, ensure_ascii=False), time.time()),
            )
            self.conn.commit()
