# -*- coding: utf-8 -*-
"""Durable evidence intake and the MemHub Core's single writer.

Every submission is stored in ``hub_writes`` before any model work, so the row is
the receipt: clients move on once it exists. One worker processes rows in event
time order (``memory_at``, else arrival), so a cold-start import from several
CLIs reaches the reconciler oldest-first. Model failures are retried with
backoff; a row only ends ``failed`` after ``max_attempts``, and ``retry_failed``
or a resubmission of the same request puts it back in the queue.

``raw_only`` is a mode, not part of the evidence: the same request can be stored
raw first and extracted later (``extract_raw``), reusing its L0 node.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from dual_mem.errors import IdempotencyConflictError

WriteProcessor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

_JOB_COLUMNS = "request_id,status,raw_only,attempts,result,error,created_at,updated_at"


class HubWriteQueue:
    """Persist source evidence before one worker serially calls MemoryClient."""

    def __init__(
        self,
        path: Path,
        process: WriteProcessor,
        *,
        max_attempts: int = 5,
        retry_base_seconds: float = 30.0,
        retry_max_seconds: float = 1800.0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS hub_writes ("
            "request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
            "payload TEXT NOT NULL, raw_only INTEGER NOT NULL, status TEXT NOT NULL, "
            "attempts INTEGER NOT NULL DEFAULT 0, next_at REAL NOT NULL DEFAULT 0, "
            "event_at INTEGER NOT NULL, result TEXT, error TEXT, "
            "created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS hub_writes_ready "
            "ON hub_writes(status, event_at)"
        )
        self._conn.execute(
            "UPDATE hub_writes SET status='pending' WHERE status='running'"
        )
        self._conn.commit()
        self._process = process
        self._max_attempts = max_attempts
        self._retry_base = retry_base_seconds
        self._retry_max = retry_max_seconds
        self._wake = asyncio.Event()
        self._closed = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="dual-mem-hub-writer")
        self._wake.set()

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Store a write once; resubmitting requeues a failed row or upgrades raw to full."""
        request_id = str(payload["request_id"])
        raw_only = bool(payload["raw_only"])
        evidence = {k: v for k, v in payload.items() if k != "raw_only"}
        encoded = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        now = int(time.time())
        row = self._conn.execute(
            "SELECT fingerprint,status,raw_only FROM hub_writes WHERE request_id=?",
            (request_id,),
        ).fetchone()
        with self._conn:
            if row is None:
                self._conn.execute(
                    "INSERT INTO hub_writes(request_id,fingerprint,payload,raw_only,"
                    "status,event_at,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        request_id,
                        fingerprint,
                        encoded,
                        int(raw_only),
                        "pending",
                        payload.get("memory_at") or now,
                        now,
                        now,
                    ),
                )
            elif row["fingerprint"] != fingerprint:
                raise IdempotencyConflictError(
                    "request_id reused with different Hub evidence"
                )
            elif row["status"] == "failed":
                self._requeue("request_id=?", (request_id,), raw_only=raw_only)
            elif row["raw_only"] and not raw_only:
                self._requeue(
                    "request_id=? AND status='done'", (request_id,), raw_only=False
                )
                self._conn.execute(
                    "UPDATE hub_writes SET raw_only=0 WHERE request_id=?",
                    (request_id,),
                )
        self._wake.set()
        return self.get(request_id)

    def _requeue(self, where: str, params: tuple, *, raw_only: bool) -> int:
        cursor = self._conn.execute(
            "UPDATE hub_writes SET status='pending',attempts=0,next_at=0,error=NULL,"
            f"raw_only=?,updated_at=? WHERE {where}",
            (int(raw_only), int(time.time()), *params),
        )
        return cursor.rowcount

    def extract_raw(self) -> int:
        """Queue extraction for every write that was stored raw-only."""
        with self._conn:
            done = self._requeue("raw_only=1 AND status='done'", (), raw_only=False)
            # Not processed yet: switching the mode is enough. A running raw job
            # sees the change when it finishes (see ``_finish``).
            self._conn.execute("UPDATE hub_writes SET raw_only=0 WHERE raw_only=1")
        self._wake.set()
        return done

    def retry_failed(self) -> int:
        with self._conn:
            count = self._conn.execute(
                "UPDATE hub_writes SET status='pending',attempts=0,next_at=0,"
                "error=NULL,updated_at=? WHERE status='failed'",
                (int(time.time()),),
            ).rowcount
        self._wake.set()
        return count

    def get(self, request_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            f"SELECT {_JOB_COLUMNS} FROM hub_writes WHERE request_id=?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return {
            "request_id": row["request_id"],
            "status": row["status"],
            "raw_only": bool(row["raw_only"]),
            "attempts": row["attempts"],
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status,COUNT(*) FROM hub_writes GROUP BY status"
        ).fetchall()
        out = {str(status): int(count) for status, count in rows}
        out["raw_only_done"] = self._conn.execute(
            "SELECT COUNT(*) FROM hub_writes WHERE raw_only=1 AND status='done'"
        ).fetchone()[0]
        return out

    async def wait(self, request_id: str) -> dict[str, Any]:
        while True:
            job = self.get(request_id)
            if job["status"] in ("done", "failed"):
                return job
            await asyncio.sleep(0.02)

    def _take(self) -> tuple[str, dict[str, Any]] | float | None:
        """The next ready job, else the time the earliest delayed one is due."""
        now = time.time()
        row = self._conn.execute(
            "SELECT request_id,payload,raw_only FROM hub_writes "
            "WHERE status='pending' AND next_at<=? ORDER BY event_at, rowid LIMIT 1",
            (now,),
        ).fetchone()
        if row is None:
            due = self._conn.execute(
                "SELECT MIN(next_at) FROM hub_writes WHERE status='pending'"
            ).fetchone()[0]
            return due
        with self._conn:
            self._conn.execute(
                "UPDATE hub_writes SET status='running',updated_at=? WHERE request_id=?",
                (int(now), row["request_id"]),
            )
        payload = json.loads(row["payload"])
        payload["raw_only"] = bool(row["raw_only"])
        return str(row["request_id"]), payload

    def _finish_ok(self, request_id: str, *, raw_only: bool, result: dict) -> None:
        with self._conn:
            # Raw evidence stored, but extraction was requested meanwhile: run again.
            self._conn.execute(
                "UPDATE hub_writes SET status=CASE WHEN ?=1 AND raw_only=0 "
                "THEN 'pending' ELSE 'done' END,result=?,error=NULL,updated_at=? "
                "WHERE request_id=?",
                (
                    int(raw_only),
                    json.dumps(result, ensure_ascii=False),
                    int(time.time()),
                    request_id,
                ),
            )

    def _finish_error(self, request_id: str, exc: Exception) -> None:
        attempts = self._conn.execute(
            "SELECT attempts FROM hub_writes WHERE request_id=?", (request_id,)
        ).fetchone()[0] + 1
        permanent = isinstance(exc, IdempotencyConflictError)
        failed = permanent or attempts >= self._max_attempts
        delay = min(self._retry_base * 2 ** (attempts - 1), self._retry_max)
        with self._conn:
            self._conn.execute(
                "UPDATE hub_writes SET status=?,attempts=?,next_at=?,error=?,"
                "updated_at=? WHERE request_id=?",
                (
                    "failed" if failed else "pending",
                    attempts,
                    0 if failed else time.time() + delay,
                    f"{type(exc).__name__}: {exc}",
                    int(time.time()),
                    request_id,
                ),
            )

    async def _run(self) -> None:
        while not self._closed:
            item = self._take()
            if not isinstance(item, tuple):
                self._wake.clear()
                timeout = None if item is None else max(item - time.time(), 0.01)
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout)
                except TimeoutError:
                    pass
                continue
            request_id, payload = item
            try:
                result = await self._process(payload)
            except Exception as exc:
                self._finish_error(request_id, exc)
            else:
                self._finish_ok(request_id, raw_only=payload["raw_only"], result=result)

    async def close(self) -> None:
        self._closed = True
        self._wake.set()
        if self._task is not None:
            await self._task
        self._conn.close()
