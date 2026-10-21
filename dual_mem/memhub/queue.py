# -*- coding: utf-8 -*-
"""Durable evidence intake and the MemHub Core's single writer."""

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


class HubWriteQueue:
    """Persist source evidence before one worker serially calls MemoryClient."""

    def __init__(self, path: Path, process: WriteProcessor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS hub_writes ("
            "request_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
            "payload TEXT NOT NULL, status TEXT NOT NULL, result TEXT, error TEXT, "
            "created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)"
        )
        self._conn.execute(
            "UPDATE hub_writes SET status='pending' WHERE status='running'"
        )
        self._conn.commit()
        self._process = process
        self._wake = asyncio.Event()
        self._closed = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="dual-mem-hub-writer")
        self._wake.set()

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(payload["request_id"])
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        row = self._conn.execute(
            "SELECT fingerprint FROM hub_writes WHERE request_id=?",
            (request_id,),
        ).fetchone()
        if row is not None:
            if row["fingerprint"] != fingerprint:
                raise IdempotencyConflictError(
                    "request_id reused with different Hub evidence"
                )
            return self.get(request_id)
        now = int(time.time())
        with self._conn:
            self._conn.execute(
                "INSERT INTO hub_writes("
                "request_id,fingerprint,payload,status,created_at,updated_at"
                ") VALUES (?,?,?,?,?,?)",
                (request_id, fingerprint, encoded, "pending", now, now),
            )
        self._wake.set()
        return self.get(request_id)

    def get(self, request_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT request_id,status,result,error,created_at,updated_at "
            "FROM hub_writes WHERE request_id=?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return {
            "request_id": row["request_id"],
            "status": row["status"],
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status,COUNT(*) FROM hub_writes GROUP BY status"
        ).fetchall()
        return {str(status): int(count) for status, count in rows}

    async def wait(self, request_id: str) -> dict[str, Any]:
        while True:
            job = self.get(request_id)
            if job["status"] in ("done", "failed"):
                return job
            await asyncio.sleep(0.02)

    def _take(self) -> tuple[str, dict[str, Any]] | None:
        row = self._conn.execute(
            "SELECT request_id,payload FROM hub_writes "
            "WHERE status='pending' ORDER BY created_at, rowid LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        with self._conn:
            self._conn.execute(
                "UPDATE hub_writes SET status='running',updated_at=? "
                "WHERE request_id=? AND status='pending'",
                (int(time.time()), row["request_id"]),
            )
        return str(row["request_id"]), json.loads(row["payload"])

    def _finish(
        self,
        request_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE hub_writes SET status=?,result=?,error=?,updated_at=? "
                "WHERE request_id=?",
                (
                    status,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    error,
                    int(time.time()),
                    request_id,
                ),
            )

    async def _run(self) -> None:
        while not self._closed:
            item = self._take()
            if item is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            request_id, payload = item
            try:
                result = await self._process(payload)
            except Exception as exc:
                self._finish(
                    request_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                self._finish(request_id, status="done", result=result)

    async def close(self) -> None:
        self._closed = True
        self._wake.set()
        if self._task is not None:
            await self._task
        self._conn.close()
