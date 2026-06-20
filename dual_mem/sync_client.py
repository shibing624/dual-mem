# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Synchronous facade over MemoryClient for scripts and sync agent hosts.

Runs all async I/O on a dedicated background event-loop thread so one client instance
can be reused across many sync calls with the same concurrency / batching behaviour as
the async API (unlike calling asyncio.run() per method).
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.sdk_models import (
    ChatMessage,
    DeleteBulkResult,
    DeleteResult,
    DigestResult,
    MemoryItem,
    ScopeSummary,
    SearchResult,
    UpdateResult,
    WriteResult,
)

_T = TypeVar("_T")


class SyncMemoryClient:
    """Blocking wrapper around :class:`MemoryClient`.

    Use this in plain scripts, REPLs, or sync web frameworks. For FastAPI / asyncio
    agents, use :class:`MemoryClient` directly.

    Reuse one instance for the process lifetime; call :meth:`close` (or use as a context
    manager) on shutdown. Do not call from inside a running event loop.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        storage_dir: str | None = None,
        mode: str | None = None,
        embed=None,
        llm=None,
    ) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="dual-mem-sync-loop",
            daemon=True,
        )
        self._thread.start()
        self._client = MemoryClient(
            settings=settings,
            storage_dir=storage_dir,
            mode=mode,
            embed=embed,
            llm=llm,
        )
        self._closed = False

    @classmethod
    def from_config(
        cls,
        config_dict: dict[str, Any],
        *,
        mode: str | None = None,
        embed=None,
        llm=None,
    ) -> SyncMemoryClient:
        """Blocking client from a mem0-style config dict (see ``MemoryClient.from_config``)."""
        settings = Settings.from_dict(config_dict)
        if mode is not None:
            settings = settings.model_copy(update={"mode": mode})
        return cls(settings=settings, embed=embed, llm=llm)

    @property
    def settings(self) -> Settings:
        return self._client.settings

    @property
    def mode(self) -> str:
        return self._client.mode

    @property
    def async_client(self) -> MemoryClient:
        """Underlying async client (advanced / bridge use)."""
        return self._client

    def __enter__(self) -> SyncMemoryClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        if self._closed:
            coro.close()
            raise RuntimeError("SyncMemoryClient is closed")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            raise RuntimeError(
                "SyncMemoryClient cannot be used inside a running event loop; "
                "use MemoryClient and await instead."
            )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _run_on_loop(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Schedule on the background loop without the caller-loop guard (shutdown only)."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def add(
        self,
        *,
        content: str = "",
        messages: list[dict] | list[ChatMessage] | None = None,
        app_id: str | None = None,
        user_id: str,
        agent_id: str = "",
        session_id: str = "",
        memory_at: int | None = None,
    ) -> WriteResult:
        return self._run(
            self._client.add(
                content=content,
                messages=messages,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_at=memory_at,
            )
        )

    def search(
        self,
        *,
        query: str,
        app_ids: list[str] | None = None,
        user_id: str,
        agent_ids: list[str] | None = None,
        session_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        created_after: int | None = None,
        request_id: str | None = None,
        debug: bool = False,
    ) -> SearchResult:
        return self._run(
            self._client.search(
                query=query,
                app_ids=app_ids,
                user_id=user_id,
                agent_ids=agent_ids,
                session_ids=session_ids,
                limit=limit,
                min_score=min_score,
                profile_limit=profile_limit,
                profile_min_score=profile_min_score,
                intention_limit=intention_limit,
                created_after=created_after,
                request_id=request_id,
                debug=debug,
            )
        )

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._run(self._client.get(memory_id))

    def list(
        self, *, app_id: str | None = None, user_id: str, agent_id: str = "", limit: int = 100
    ) -> list[MemoryItem]:
        return self._run(
            self._client.list(
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                limit=limit,
            )
        )

    def update(self, memory_id: str, content: str) -> UpdateResult:
        return self._run(self._client.update(memory_id, content))

    def delete(self, memory_id: str) -> DeleteResult:
        return self._run(self._client.delete(memory_id))

    def delete_bulk(
        self,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        confirm: bool = False,
    ) -> DeleteBulkResult:
        return self._run(
            self._client.delete_bulk(
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                confirm=confirm,
            )
        )

    def list_scopes(
        self,
        *,
        app_id: str | None = None,
        limit: int = 5000,
    ) -> list[ScopeSummary]:
        return self._run(self._client.list_scopes(app_id=app_id, limit=limit))

    def digest(self) -> DigestResult:
        return self._run(self._client.digest())

    def close(self) -> None:
        """Release resources; idempotent. Prefer ``with SyncMemoryClient(...) as client:``."""
        if self._closed:
            return
        try:
            self._run_on_loop(self._client.aclose())
        finally:
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
