# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SQLite-backed cache store holding profile caches, the System2/intention
task queues, pipeline logs and a memory-operation audit trail.
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
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    app_id  TEXT NOT NULL,
    status  TEXT NOT NULL DEFAULT 'pending',
    ts      REAL NOT NULL
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
"""


class CacheStore:
    """SQLite store for profile caches, async task queues, pipeline logs and op records."""

    def __init__(self, storage_dir: str):
        self.conn = sqlite3.connect(f"{storage_dir}/cache.db", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_DDL)
        self.conn.commit()

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

    def enqueue_s2_task(self, user_id: str, app_id: str) -> None:
        """Append a pending System2 distillation task for an app/user pair."""
        self.conn.execute(
            "INSERT INTO s2_queue (user_id, app_id, status, ts) VALUES (?, ?, 'pending', ?)",
            (user_id, app_id, time.time()),
        )
        self.conn.commit()

    def dequeue_s2_task(self) -> dict | None:
        """Pop the oldest pending System2 task, marking it done; None if queue empty."""
        row = self.conn.execute(
            "SELECT * FROM s2_queue WHERE status = 'pending' ORDER BY id ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        self.conn.execute(
            "UPDATE s2_queue SET status = 'done' WHERE id = ?", (row["id"],)
        )
        self.conn.commit()
        task = dict(row)
        task["status"] = "done"
        return task

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
