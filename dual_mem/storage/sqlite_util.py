# -*- coding: utf-8 -*-
"""Shared SQLite connection helpers."""

from __future__ import annotations

import sqlite3


def connect_sqlite(path: str) -> sqlite3.Connection:
    """Open SQLite with WAL + NORMAL sync (better concurrent read/write on one file)."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn
