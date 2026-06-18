import json
import sqlite3
import time

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
    def __init__(self, storage_dir: str):
        self.conn = sqlite3.connect(
            f"{storage_dir}/history.db", check_same_thread=False
        )
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
