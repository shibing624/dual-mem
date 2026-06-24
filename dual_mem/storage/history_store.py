# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SQLite-backed append-only history store recording memory lifecycle events
(add/update/delete) with old/new snapshots for auditing.
"""
import json
import sqlite3
import threading
import time

from dual_mem.storage.sqlite_util import connect_sqlite

_DDL = """
CREATE TABLE IF NOT EXISTS history (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    event   TEXT NOT NULL,
    node_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    old     TEXT,
    new     TEXT,
    ts      REAL NOT NULL
);
"""


class HistoryStore:
    """Append-only audit log of memory lifecycle events."""

    def __init__(self, storage_dir: str, *, persist: bool = True):
        self._persist = persist
        self._lock = threading.RLock()
        self.conn = None
        if persist:
            with self._lock:
                self.conn = connect_sqlite(f"{storage_dir}/history.db")
                self.conn.row_factory = sqlite3.Row
                self.conn.executescript(_DDL)
                self.conn.commit()

    def append(
        self,
        *,
        event: str,
        node_id: str,
        user_id: str,
        old: dict | None,
        new: dict | None,
    ) -> None:
        """Append one history event with optional old/new metadata snapshots."""
        if not self._persist or self.conn is None:
            return
        with self._lock:
            self.conn.execute(
                "INSERT INTO history (event, node_id, user_id, old, new, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event,
                    node_id,
                    user_id,
                    json.dumps(old, ensure_ascii=False) if old is not None else None,
                    json.dumps(new, ensure_ascii=False) if new is not None else None,
                    time.time(),
                ),
            )
            self.conn.commit()

    def list_for_node(self, node_id: str) -> list[dict]:
        """Return all history events for a node in chronological order, snapshots decoded."""
        if not self._persist or self.conn is None:
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM history WHERE node_id = ? ORDER BY id ASC", (node_id,)
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["old"] = json.loads(item["old"]) if item["old"] is not None else None
                item["new"] = json.loads(item["new"]) if item["new"] is not None else None
                result.append(item)
            return result
