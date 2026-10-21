# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MemHub home (~/.dual-mem): Core connection, local scan cursors and
the Core-owned memory store.

CLI hosts never open the store. They read local transcripts and send them to one
Hub Core over loopback HTTP. The Core is the only MemoryClient writer.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dual_mem.config import Settings

DEFAULT_HUB_DIR = Path.home() / ".dual-mem"


@dataclass
class HubConfig:
    """Optional remote Core attachment persisted as ``<hub>/hub.json``."""

    base_url: str = ""
    api_key: str = ""

    def resolved_base_url(self) -> str:
        return (os.environ.get("DUAL_MEM_HUB_BASE_URL") or self.base_url).rstrip("/")

    def resolved_api_key(self) -> str:
        return os.environ.get("DUAL_MEM_HUB_API_KEY") or self.api_key

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["api_key"] = "***" if data["api_key"] else ""
        return data


def hub_root(hub_dir: str | Path | None = None) -> Path:
    """``hub_dir`` > ``DUAL_MEM_HUB_DIR`` > ``~/.dual-mem``."""
    raw = hub_dir or os.environ.get("DUAL_MEM_HUB_DIR")
    path = Path(raw).expanduser().resolve() if raw else DEFAULT_HUB_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def hub_store_dir(hub_dir: str | Path | None = None) -> Path:
    return hub_root(hub_dir) / "store"


def hub_upload_dir(hub_dir: str | Path | None = None) -> Path:
    path = hub_root(hub_dir) / "uploads"
    path.mkdir(exist_ok=True)
    return path


def _config_path(hub_dir: str | Path | None) -> Path:
    return hub_root(hub_dir) / "hub.json"


def _runtime_path(hub_dir: str | Path | None) -> Path:
    return hub_root(hub_dir) / "runtime.json"


def _token_path(hub_dir: str | Path | None) -> Path:
    return hub_root(hub_dir) / "runtime-token"


def save_hub_config(cfg: HubConfig, hub_dir: str | Path | None = None) -> Path:
    path = _config_path(hub_dir)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def clear_hub_config(hub_dir: str | Path | None = None) -> bool:
    """Remove a remote attachment so clients resume local auto-discovery."""
    path = _config_path(hub_dir)
    if not path.is_file():
        return False
    path.unlink()
    return True


def load_hub_config(hub_dir: str | Path | None = None) -> HubConfig:
    """Read an explicit remote attachment, returning empty defaults if absent."""
    path = _config_path(hub_dir)
    if not path.is_file():
        return HubConfig()
    return HubConfig(**json.loads(path.read_text(encoding="utf-8")))


def load_or_create_runtime_token(
    hub_dir: str | Path | None = None,
    *,
    token: str | None = None,
) -> str:
    """Return the Core bearer token, creating its 0600 file once."""
    path = _token_path(hub_dir)
    if token is None and path.is_file():
        return path.read_text(encoding="utf-8").strip()
    value = token or secrets.token_urlsafe(32)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value + "\n")
    os.replace(tmp, path)
    return value


def publish_runtime(
    *,
    base_url: str,
    instance_id: str,
    hub_dir: str | Path | None = None,
) -> None:
    """Publish the running local Core for CLI/MCP auto-discovery."""
    path = _runtime_path(hub_dir)
    tmp = path.with_suffix(".tmp")
    payload = {
        "base_url": base_url.rstrip("/"),
        "instance_id": instance_id,
        "pid": os.getpid(),
        "started_at": int(time.time()),
    }
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def unpublish_runtime(
    instance_id: str,
    hub_dir: str | Path | None = None,
) -> None:
    """Remove only this Core's receipt, never a replacement process's receipt."""
    path = _runtime_path(hub_dir)
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["instance_id"] == instance_id:
        path.unlink()


def resolve_hub_config(hub_dir: str | Path | None = None) -> HubConfig:
    """Explicit remote attachment, env pair, or local runtime discovery."""
    configured = load_hub_config(hub_dir)
    base_url = configured.resolved_base_url()
    api_key = configured.resolved_api_key()
    if bool(base_url) != bool(api_key):
        raise ValueError("Hub base_url and api_key must be configured together")
    if base_url:
        return HubConfig(base_url=base_url, api_key=api_key)

    runtime_path = _runtime_path(hub_dir)
    token_path = _token_path(hub_dir)
    if not runtime_path.is_file() or not token_path.is_file():
        raise RuntimeError(
            "local Hub Core is not running; start `dual-mem hub serve` "
            "or configure `dual-mem hub attach`"
        )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    return HubConfig(
        base_url=str(runtime["base_url"]),
        api_key=token_path.read_text(encoding="utf-8").strip(),
    )


def runtime_status(hub_dir: str | Path | None = None) -> dict[str, Any] | None:
    path = _runtime_path(hub_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def hub_settings(hub_dir: str | Path | None = None, *, mode: str | None = None) -> Settings:
    """Settings from the usual sources, with storage pinned to the hub store."""
    overrides: dict[str, Any] = {"storage_dir": str(hub_store_dir(hub_dir))}
    if mode is not None:
        overrides["mode"] = mode
    return Settings(**overrides)


@dataclass(frozen=True)
class Cursor:
    source: str
    path: str
    # Byte offset for JSONL transcripts, last message id for Hermes' SQLite.
    position: int
    turns: int
    updated_at: int


class CursorStore:
    """Where each transcript was last read up to, plus a per-transcript scan lock."""

    def __init__(self, hub_dir: str | Path | None = None) -> None:
        self._root = hub_root(hub_dir)
        self._conn = sqlite3.connect(str(self._root / "cursors.db"), timeout=30)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cursors (source TEXT NOT NULL, path TEXT NOT NULL, "
            "position INTEGER NOT NULL, turns INTEGER NOT NULL, updated_at INTEGER NOT NULL, "
            "PRIMARY KEY (source, path))"
        )
        self._conn.commit()

    def __enter__(self) -> CursorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self._conn.close()

    def get(self, source: str, path: str) -> Cursor | None:
        row = self._conn.execute(
            "SELECT source, path, position, turns, updated_at FROM cursors "
            "WHERE source=? AND path=?",
            (source, path),
        ).fetchone()
        return Cursor(*row) if row else None

    def advance(self, source: str, path: str, position: int, turns: int) -> None:
        """Move the cursor to ``position`` and add ``turns`` to its ingested count."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO cursors (source, path, position, turns, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(source, path) DO UPDATE SET "
                "position=excluded.position, turns=cursors.turns + excluded.turns, "
                "updated_at=excluded.updated_at",
                (source, path, position, turns, int(time.time())),
            )

    def all(self) -> list[Cursor]:
        rows = self._conn.execute(
            "SELECT source, path, position, turns, updated_at FROM cursors "
            "ORDER BY updated_at DESC"
        ).fetchall()
        return [Cursor(*row) for row in rows]

    @contextmanager
    def lock(self, source: str, path: str) -> Iterator[None]:
        """Serialize scanners of one transcript (two Stop hooks can overlap)."""
        locks = self._root / "locks"
        locks.mkdir(exist_ok=True)
        name = hashlib.sha1(f"{source}\0{path}".encode()).hexdigest()
        with (locks / f"{name}.lock").open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
