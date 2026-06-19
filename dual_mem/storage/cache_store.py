# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SQLite-backed cache store holding profile caches, the System2/intention/reconcile
task queues, pipeline logs, a memory-operation audit trail and read-side access counters.
"""
import json
import sqlite3
import time

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
    task_type TEXT NOT NULL DEFAULT 'cognition',
    payload   TEXT NOT NULL DEFAULT '',
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
CREATE TABLE IF NOT EXISTS memory_access (
    node_id          TEXT PRIMARY KEY,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at REAL    NOT NULL
);
"""


class CacheStore:
    """SQLite store for profile caches, async task queues, pipeline logs, op records and access counters."""

    def __init__(self, storage_dir: str):
        self.conn = sqlite3.connect(f"{storage_dir}/cache.db", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_DDL)
        self._migrate_s2_queue()
        self.conn.commit()

    def _migrate_s2_queue(self) -> None:
        """Add task_type/payload columns to s2_queue when upgrading from older schemas."""
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(s2_queue)").fetchall()}
        if "task_type" not in cols:
            self.conn.execute(
                "ALTER TABLE s2_queue ADD COLUMN task_type TEXT NOT NULL DEFAULT 'cognition'"
            )
        if "payload" not in cols:
            self.conn.execute(
                "ALTER TABLE s2_queue ADD COLUMN payload TEXT NOT NULL DEFAULT ''"
            )

    # ---- Profile cache --------------------------------------------------------------

    def set_profile(self, iso_key: str, data: dict) -> None:
        """Store (or replace) the cached profile for an isolation key."""
        self.conn.execute(
            "INSERT OR REPLACE INTO profile_cache (iso_key, data, ts) VALUES (?, ?, ?)",
            (iso_key, json.dumps(data, ensure_ascii=False), time.time()),
        )
        self.conn.commit()

    def get_profile(self, iso_key: str) -> dict | None:
        """Return the cached profile for an isolation key, or None if missing."""
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
        task_type: str = "cognition",
        payload: dict | None = None,
    ) -> None:
        """Append a pending S2 task; deduped per (app, user, agent, task_type)."""
        existing = self.conn.execute(
            "SELECT 1 FROM s2_queue WHERE status = 'pending' "
            "AND user_id = ? AND app_id = ? AND agent_id = ? AND task_type = ? LIMIT 1",
            (user_id, app_id, agent_id, task_type),
        ).fetchone()
        if existing is not None:
            return
        payload_str = json.dumps(payload, ensure_ascii=False) if payload is not None else ""
        self.conn.execute(
            "INSERT INTO s2_queue (user_id, app_id, agent_id, status, task_type, payload, ts) "
            "VALUES (?, ?, ?, 'pending', ?, ?, ?)",
            (user_id, app_id, agent_id, task_type, payload_str, time.time()),
        )
        self.conn.commit()

    def dequeue_s2_task(self, task_type: str | None = None) -> dict | None:
        """Pop the oldest pending System2 task (FIFO), marking it done; None if queue empty.

        ``task_type`` filters the queue (e.g. ``"reconsolidation"``) so callers can drain a
        specific class of work without touching cognition tasks.
        """
        if task_type is None:
            row = self.conn.execute(
                "SELECT * FROM s2_queue WHERE status = 'pending' ORDER BY id ASC LIMIT 1"
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM s2_queue WHERE status = 'pending' AND task_type = ? "
                "ORDER BY id ASC LIMIT 1",
                (task_type,),
            ).fetchone()
        if row is None:
            return None
        self.conn.execute(
            "UPDATE s2_queue SET status = 'done' WHERE id = ?", (row["id"],)
        )
        self.conn.commit()
        task = dict(row)
        task["status"] = "done"
        raw_payload = task.get("payload") or ""
        task["payload"] = json.loads(raw_payload) if raw_payload else {}
        return task

    def list_pending_s2_users(self) -> list[dict]:
        """Snapshot the unique (app, user, agent, task_type) tuples currently pending in the queue."""
        rows = self.conn.execute(
            "SELECT DISTINCT app_id, user_id, agent_id, task_type "
            "FROM s2_queue WHERE status = 'pending'"
        ).fetchall()
        return [dict(row) for row in rows]

    # ---- Reconcile queue (asynchronous evolution chain merging) ---------------------

    def enqueue_reconcile_task(
        self, *, app_id: str, user_id: str, agent_id: str, node_ids: list[str]
    ) -> None:
        """Append a reconcile task carrying the freshly written node ids that need merging."""
        if not node_ids:
            return
        self.conn.execute(
            "INSERT INTO reconcile_queue (app_id, user_id, agent_id, node_ids, status, ts) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (app_id, user_id, agent_id, json.dumps(node_ids, ensure_ascii=False), time.time()),
        )
        self.conn.commit()

    def dequeue_reconcile_task(
        self, *, app_id: str | None = None, user_id: str | None = None, agent_id: str | None = None
    ) -> dict | None:
        """Pop the oldest pending reconcile task, optionally scoped to an (app/user/agent)."""
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
        sql = f"SELECT * FROM reconcile_queue WHERE {' AND '.join(clauses)} ORDER BY id ASC LIMIT 1"
        row = self.conn.execute(sql, params).fetchone()
        if row is None:
            return None
        self.conn.execute(
            "UPDATE reconcile_queue SET status = 'done' WHERE id = ?", (row["id"],)
        )
        self.conn.commit()
        task = dict(row)
        task["node_ids"] = json.loads(task["node_ids"]) if task["node_ids"] else []
        task["status"] = "done"
        return task

    def reconcile_queue_size(self) -> int:
        """Return how many reconcile tasks are still pending across all users."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM reconcile_queue WHERE status = 'pending'"
        ).fetchone()
        return int(row["n"]) if row else 0

    # ---- Pipeline logs --------------------------------------------------------------

    def log_pipeline(self, *, request_id: str, stage: str, payload: dict) -> None:
        """Append a structured pipeline-stage log entry for a request."""
        self.conn.execute(
            "INSERT INTO pipeline_logs (request_id, stage, payload, ts) VALUES (?, ?, ?, ?)",
            (request_id, stage, json.dumps(payload, ensure_ascii=False), time.time()),
        )
        self.conn.commit()

    def list_pipeline_logs(self, request_id: str) -> list[dict]:
        """Return all pipeline log entries for a request in order, payloads decoded."""
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
        self.conn.execute(
            "INSERT INTO memory_operations (op, node_id, user_id, ts) VALUES (?, ?, ?, ?)",
            (op, node_id, user_id, time.time()),
        )
        self.conn.commit()

    # ---- Read-side access counters (used by Reconsolidation Hook) -------------------

    def bump_access(self, node_ids: list[str]) -> None:
        """Increment access_count and refresh last_accessed_at for each node id (no-op on empty)."""
        if not node_ids:
            return
        now = time.time()
        for nid in node_ids:
            self.conn.execute(
                "INSERT INTO memory_access (node_id, access_count, last_accessed_at) "
                "VALUES (?, 1, ?) ON CONFLICT(node_id) DO UPDATE SET "
                "access_count = access_count + 1, last_accessed_at = excluded.last_accessed_at",
                (nid, now),
            )
        self.conn.commit()

    def get_access(self, node_id: str) -> dict | None:
        """Return {access_count, last_accessed_at} for a node, or None if never accessed."""
        row = self.conn.execute(
            "SELECT access_count, last_accessed_at FROM memory_access WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        return dict(row) if row else None
