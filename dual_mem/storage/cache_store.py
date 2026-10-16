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
CREATE TABLE IF NOT EXISTS request_outcomes (
    request_key TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    ts          REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS review_queue (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind      TEXT NOT NULL,
    app_id    TEXT NOT NULL,
    user_id   TEXT NOT NULL,
    agent_id  TEXT NOT NULL DEFAULT '',
    node_id   TEXT NOT NULL,
    payload   TEXT NOT NULL,
    status    TEXT NOT NULL DEFAULT 'pending',
    attempts  INTEGER NOT NULL DEFAULT 0,
    error     TEXT NOT NULL DEFAULT '',
    ts        REAL NOT NULL,
    UNIQUE(kind, node_id)
);
CREATE TABLE IF NOT EXISTS s2_failures (
    fact_id   TEXT NOT NULL,
    scope     TEXT NOT NULL,
    attempts  INTEGER NOT NULL DEFAULT 0,
    error     TEXT NOT NULL DEFAULT '',
    ts        REAL NOT NULL,
    PRIMARY KEY (scope, fact_id)
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

    # ---- Idempotent write outcomes --------------------------------------------------

    def set_request_outcome(self, request_key: str, data: dict) -> None:
        """Persist the outcome of an idempotent add(request_id=...)."""
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO request_outcomes (request_key, data, ts) VALUES (?, ?, ?)",
                (request_key, json.dumps(data, ensure_ascii=False), time.time()),
            )
            self.conn.commit()

    def get_request_outcome(self, request_key: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT data FROM request_outcomes WHERE request_key = ?",
                (request_key,),
            ).fetchone()
            return json.loads(row["data"]) if row else None

    def purge_request_outcomes(self, *, older_than: float) -> int:
        """Drop outcomes older than ``older_than`` (unix seconds); returns rows removed."""
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM request_outcomes WHERE ts < ?", (older_than,)
            )
            self.conn.commit()
            return cur.rowcount

    # ---- Review queue (undecided relations, sources needing re-extraction) ---------

    def enqueue_review(
        self,
        *,
        kind: str,
        app_id: str,
        user_id: str,
        agent_id: str,
        node_id: str,
        payload: dict,
    ) -> None:
        """Queue one review item; re-queuing the same (kind, node) reopens it."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO review_queue (kind, app_id, user_id, agent_id, node_id, payload, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(kind, node_id) DO UPDATE SET payload=excluded.payload, "
                "status='pending', attempts=0, error='', ts=excluded.ts",
                (
                    kind,
                    app_id,
                    user_id,
                    agent_id,
                    node_id,
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )
            self.conn.commit()

    def list_reviews(self, *, status: str = "pending") -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM review_queue WHERE status = ? ORDER BY id ASC",
                (status,),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"])
            items.append(item)
        return items

    def resolve_review(self, review_id: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE review_queue SET status='done' WHERE id=?", (review_id,)
            )
            self.conn.commit()

    def fail_review(self, review_id: int, *, error: str, max_attempts: int) -> None:
        """Record a failed review; quarantine it after ``max_attempts``."""
        with self._lock:
            self.conn.execute(
                "UPDATE review_queue SET attempts=attempts+1, error=?, "
                "status=CASE WHEN attempts+1 >= ? THEN 'quarantined' ELSE 'pending' END "
                "WHERE id=?",
                (error[:500], max_attempts, review_id),
            )
            self.conn.commit()

    # ---- System2 failed evidence ----------------------------------------------------

    def record_s2_failure(
        self, *, scope: str, fact_ids: list[str], error: str, increment: int = 1
    ) -> None:
        """Count a failed cognition batch against each fact it contained."""
        with self._lock:
            self.conn.executemany(
                "INSERT INTO s2_failures (fact_id, scope, attempts, error, ts) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(scope, fact_id) DO UPDATE SET "
                "attempts=attempts+excluded.attempts, error=excluded.error, ts=excluded.ts",
                [(fid, scope, increment, error[:500], time.time()) for fid in fact_ids],
            )
            self.conn.commit()

    def clear_s2_failures(self, *, scope: str, fact_ids: list[str]) -> None:
        with self._lock:
            self.conn.executemany(
                "DELETE FROM s2_failures WHERE scope=? AND fact_id=?",
                [(scope, fid) for fid in fact_ids],
            )
            self.conn.commit()

    def quarantined_s2_facts(self, *, scope: str, max_attempts: int) -> set[str]:
        """Facts that failed cognition ``max_attempts`` times; excluded from batches."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT fact_id FROM s2_failures WHERE scope=? AND attempts >= ?",
                (scope, max_attempts),
            ).fetchall()
        return {row["fact_id"] for row in rows}

    def list_s2_failures(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM s2_failures ORDER BY ts ASC"
            ).fetchall()
        return [dict(row) for row in rows]

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
