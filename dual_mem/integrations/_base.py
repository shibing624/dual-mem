# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 集成层基础组件。

- ``AsyncRunner``     : 在独立后台事件循环线程里驱动异步 MemoryClient，供 Hermes /
                        OpenClaw 的同步 hook 回调安全调用。
- ``MemoryBackend``   : 与生态无关的 MemoryClient 薄封装（add/search/get/list/...
                        + render_context）。
- ``format_memories_for_prompt`` : 借鉴 hy_memory 的 <relevant-memories> 注入块，
                        支持演化链展开与长度截断。
- ``_SyncMemoryProvider`` : Hermes / OpenClaw provider 的共享基类（生命周期、写入
                        节流、线程池、工具分发）。
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterable, Optional

from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.sdk_models import MemoryItem

logger = logging.getLogger("dual_mem.integrations")

# 过短的确认类 query 不去搜记忆（与 hy_memory 一致）。
_SKIP_QUERIES = {
    "ok", "好", "好的", "thanks", "谢谢", "y", "n", "yes", "no",
    "继续", "go", "嗯", "嗯嗯", "对", "对的",
}


class AsyncRunner:
    """Run an async MemoryClient from synchronous plugin callbacks.

    A dedicated daemon thread owns the event loop; ``run(coro)`` schedules the
    coroutine on that loop and blocks until it resolves. Mirrors the ``_LoopThread``
    pattern used by hy_memory so sync host runtimes (Hermes / OpenClaw) can call
    async memory operations without managing their own loop.
    """

    def __init__(self, client: MemoryClient) -> None:
        self._client = client
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="dual-mem-loop"
        )
        self._thread.start()
        logger.debug("[runner] loop thread started, is_running=%s", self._loop.is_running())

    def run(self, coro: Any) -> Any:
        logger.debug("[runner] submit coroutine")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        result = future.result()
        logger.debug("[runner] coroutine done")
        return result

    def close(self) -> None:
        if not self._loop.is_running():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


def _fmt_time(ts: Any) -> str:
    """Unix 秒 → 'YYYY-MM-DD HH:MM'；无效返回 ''。"""
    if ts is None:
        return ""
    try:
        return datetime.datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, OverflowError, OSError):
        return ""


def format_memories_for_prompt(
    items: Iterable[MemoryItem],
    *,
    max_chars: int = 2000,
) -> str:
    """把召回结果格式化为 system prompt 注入块（借鉴 hy_memory 的 memoryContext）。

    - 整块包在 ``<relevant-memories>…</relevant-memories>`` 内。
    - 普通记忆：``- [N] <time>  <content>``
    - 演化链（chain 长度>1，index 0 = 最新）：展开为 旧→新：
      ``- [N] [Evolved, K versions]`` + 各级 ``[vK] …`` / ``[Latest] …``
    - 总长度截断到 ``max_chars``，避免一次灌爆 system prompt。
    """
    items = list(items)
    if not items:
        return ""

    out: list[str] = []
    running = 0
    idx = 0
    for mem in items:
        chain = mem.evolution_chain
        if chain and len(chain) > 1:
            lines: list[str] = []
            for i in range(len(chain) - 1, 0, -1):
                c = chain[i]
                when = _fmt_time(c.memory_at)
                lines.append(
                    f"  [v{len(chain) - i}] {when + '  ' if when else ''}{(c.content or '').strip()}"
                )
            head = chain[0]
            head_when = _fmt_time(head.memory_at)
            head_content = (head.content or mem.content or "").strip()
            lines.append(f"  [Latest] {head_when + '  ' if head_when else ''}{head_content}")
            entry = f"- [{idx + 1}] [Evolved, {len(chain)} versions]\n" + "\n".join(lines)
        else:
            content = (mem.content or "").strip()
            if not content:
                continue
            when = _fmt_time(mem.memory_at)
            entry = f"- [{idx + 1}] {when + '  ' if when else ''}{content}"

        if len(entry) > 800:
            entry = entry[:800].rstrip() + "..."
        if running + len(entry) > max_chars:
            break
        out.append(entry)
        running += len(entry) + 1
        idx += 1

    if not out:
        return ""
    body = "\n".join(out)
    return (
        "<relevant-memories>\n"
        "The following are stored memories for the current user. Use them to "
        "personalize your response. Memories with evolution chains are expanded "
        "from oldest to newest:\n"
        f"{body}\n"
        "</relevant-memories>"
    )


class MemoryBackend:
    """Ecosystem-agnostic facade over MemoryClient, with prompt rendering."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        client: Optional[MemoryClient] = None,
        storage_dir: Optional[str] = None,
        mode: Optional[str] = None,
        embed: Any = None,
        llm: Any = None,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            if settings is None:
                overrides: dict[str, Any] = {}
                if storage_dir is not None:
                    overrides["storage_dir"] = storage_dir
                if mode is not None:
                    overrides["mode"] = mode
                settings = Settings(**overrides)
            self.client = MemoryClient(settings=settings, embed=embed, llm=llm)

    async def add(
        self,
        *,
        user_id: str,
        app_id: Optional[str] = None,
        content: str = "",
        messages: Optional[list] = None,
        agent_id: str = "",
        session_id: str = "",
        memory_at: Optional[int] = None,
    ) -> Any:
        return await self.client.add(
            content=content,
            messages=messages,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            memory_at=memory_at,
        )

    async def search(
        self,
        *,
        query: str,
        user_id: str,
        app_ids: Optional[list[str]] = None,
        agent_ids: Optional[list[str]] = None,
        session_ids: Optional[list[str]] = None,
        limit: int = 10,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        created_after: Optional[int] = None,
    ) -> Any:
        return await self.client.search(
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
        )

    async def get(self, memory_id: str) -> Any:
        return await self.client.get(memory_id)

    async def list(
        self, *, user_id: str, app_id: Optional[str] = None,
        agent_id: str = "", limit: int = 100,
    ) -> list:
        return await self.client.list(
            app_id=app_id, user_id=user_id, agent_id=agent_id, limit=limit
        )

    async def update(self, memory_id: str, content: str) -> Any:
        return await self.client.update(memory_id, content)

    async def delete(self, memory_id: str) -> Any:
        return await self.client.delete(memory_id)

    async def delete_scope(
        self, *, confirm: bool, app_id: Optional[str] = None,
        user_id: Optional[str] = None, agent_id: Optional[str] = None,
    ) -> Any:
        return await self.client.delete_bulk(
            app_id=app_id, user_id=user_id, agent_id=agent_id, confirm=confirm
        )

    async def digest(self) -> Any:
        return await self.client.digest()

    async def aclose(self) -> None:
        await self.client.aclose()

    async def render_context(
        self,
        *,
        query: str,
        user_id: str,
        app_ids: Optional[list[str]] = None,
        agent_ids: Optional[list[str]] = None,
        limit: int = 10,
        max_chars: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Search, then render the matched memories as a prompt injection block."""
        result = await self.search(
            query=query, user_id=user_id, app_ids=app_ids,
            agent_ids=agent_ids, limit=limit, **kwargs,
        )
        items = result.memories.flatten(limit=limit)
        return format_memories_for_prompt(items, max_chars=max_chars)


# ============================================================================
# 共享同步 Provider 基类（Hermes / OpenClaw 同源 MemoryProvider 契约）
# ============================================================================

class _SyncMemoryProvider:
    """Common lifecycle for agent-framework memory providers.

    Subclasses only override ``name`` / config wiring / ``register``; the shared
    machinery handles the embedded client, write turn-window buffering, the sync
    thread pool, prefetch formatting and LLM tool dispatch.
    """

    name: str = "dual-mem"

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        storage_dir: Optional[str] = None,
        mode: Optional[str] = None,
        write_turn_window: int = 5,
        max_prefetch_chars: int = 2000,
        agent_id: str = "",
        user_id: str = "",
        **client_kwargs: Any,
    ) -> None:
        self._settings = settings
        self._storage_dir = storage_dir
        self._mode = mode
        self._client_kwargs = client_kwargs
        self._user_id = user_id
        self._agent_id = agent_id or "dual-mem"
        self._session_id = "default_session"
        self._write_turn_window = max(1, int(write_turn_window))
        self._max_prefetch_chars = max(200, int(max_prefetch_chars))
        self._initialized = False
        self._lock = threading.RLock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._inflight: list[Future] = []
        self._turn_buffer: dict[str, list[dict]] = {}
        self._buffer_lock = threading.Lock()
        self._backend: Optional[MemoryBackend] = None
        self._runner: Optional[AsyncRunner] = None

    # ---- 子类可实现：可用性 / 初始化 --------------------------------------
    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        with self._lock:
            if self._initialized:
                return
            self._session_id = session_id or self._session_id
            backend = MemoryBackend(
                settings=self._settings, storage_dir=self._storage_dir,
                mode=self._mode, **self._client_kwargs,
            )
            logger.debug("[provider] backend built")
            self._backend = backend
            self._runner = AsyncRunner(backend.client)
            logger.debug("[provider] runner built")
            self._executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="dual-mem-sync"
            )
            self._initialized = True

    # ---- 上下文注入（被动） ----------------------------------------------
    def prefetch(self, query: str, **kwargs: Any) -> str:
        if not self._backend or not self._runner or not query:
            return ""
        q = query.strip()
        if len(q) < 3 or q.lower() in _SKIP_QUERIES:
            return ""
        try:
            result = self._runner.run(
                self._backend.search(
                    query=q, user_id=self._user_id,
                    agent_ids=[self._agent_id], limit=10,
                )
            )
            items = result.memories.flatten(limit=10)
            if not items:
                return ""
            return format_memories_for_prompt(items, max_chars=self._max_prefetch_chars)
        except Exception as exc:  # prefetch must never block the host session
            logger.warning("[%s] prefetch failed: %s", self.name, exc)
            return ""

    def queue_prefetch(self, query: str, **kwargs: Any) -> None:
        return

    def system_prompt_block(self) -> str:
        return (
            "You have access to dual-mem — a persistent layered memory that "
            "remembers user preferences, facts, and context across sessions. "
            "Relevant memories are automatically provided before each response."
        )

    # ---- 写入节流（sync_turn） -------------------------------------------
    def sync_turn(self, user_message: str, assistant_response: str, **kwargs: Any) -> None:
        if not self._backend or not self._executor or not user_message:
            return
        session_id = kwargs.get("session_id") or self._session_id or "default_session"
        with self._buffer_lock:
            buf = self._turn_buffer.setdefault(session_id, [])
            buf.append({"role": "user", "content": user_message})
            buf.append({"role": "assistant", "content": assistant_response or ""})
            turns = sum(1 for m in buf if m["role"] == "user")
            if turns < self._write_turn_window:
                return
            batch = buf[:]
            self._turn_buffer[session_id] = []
        self._submit_write(batch, session_id)

    def _submit_write(self, messages: list[dict], session_id: str) -> None:
        if not messages or self._executor is None:
            return
        try:
            fut = self._executor.submit(self._do_sync_turn, messages, session_id)
        except RuntimeError:
            return
        self._inflight.append(fut)
        self._inflight = [f for f in self._inflight if not f.done()]

    def _flush_session_buffer(self, session_id: Optional[str] = None) -> None:
        with self._buffer_lock:
            if session_id is None:
                pending = [(sid, msgs[:]) for sid, msgs in self._turn_buffer.items() if msgs]
                self._turn_buffer.clear()
            else:
                msgs = self._turn_buffer.get(session_id) or []
                pending = [(session_id, msgs[:])] if msgs else []
                if session_id in self._turn_buffer:
                    self._turn_buffer[session_id] = []
        for sid, msgs in pending:
            self._submit_write(msgs, sid)

    def _do_sync_turn(self, messages: list[dict], session_id: str) -> None:
        assert self._backend is not None and self._runner is not None
        try:
            self._runner.run(
                self._backend.add(
                    messages=messages, user_id=self._user_id,
                    agent_id=self._agent_id, session_id=session_id,
                )
            )
        except Exception as exc:
            logger.warning("[%s] sync_turn failed: %s", self.name, exc)

    # ---- 会话结束 / 压缩前 ----------------------------------------------
    def on_session_end(self, messages: list[dict], **kwargs: Any) -> None:
        if not self._backend:
            return
        self._flush_session_buffer(kwargs.get("session_id"))
        self._wait_inflight()

    def on_pre_compress(self, messages: list[dict], **kwargs: Any) -> None:
        self.on_session_end(messages, **kwargs)

    def on_memory_write(self, action: str, target: str, content: str, **kwargs: Any) -> None:
        if not self._backend or not self._runner or not content:
            return
        if action == "delete":
            return
        try:
            self._runner.run(
                self._backend.add(
                    content=content, user_id=self._user_id,
                    agent_id=self._agent_id, session_id=self._session_id,
                )
            )
        except Exception as exc:
            logger.warning("[%s] on_memory_write failed: %s", self.name, exc)

    # ---- LLM 工具分发 ----------------------------------------------------
    def get_tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "memory_search",
                "description": "Search stored memories for relevant context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results (default 10)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_add",
                "description": "Store a new memory for future reference.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "memory_delete",
                "description": "Delete a memory by its memory_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "memory_list",
                "description": "List stored memories for the current user/agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Max results (default 20)"},
                    },
                },
            },
        ]

    def handle_tool_call(self, name: str, args: dict) -> str:
        import json
        return json.dumps(self._dispatch_tool(name, args), ensure_ascii=False)

    def _dispatch_tool(self, name: str, args: dict) -> dict:
        if not self._backend or not self._runner:
            return {"error": "Provider not initialized"}
        try:
            if name == "memory_search":
                result = self._runner.run(
                    self._backend.search(
                        query=args.get("query", ""), user_id=self._user_id,
                        agent_ids=[self._agent_id], limit=int(args.get("limit") or 10),
                    )
                )
                items = result.memories.flatten(limit=int(args.get("limit") or 10))
                return {
                    "status": "success",
                    "count": len(items),
                    "memories": [
                        {"memory_id": m.memory_id, "content": m.content, "score": m.score}
                        for m in items
                    ],
                }
            if name == "memory_add":
                content = args.get("content", "")
                if not content:
                    return {"error": "content is required"}
                res = self._runner.run(
                    self._backend.add(
                        content=content, user_id=self._user_id,
                        agent_id=self._agent_id, session_id=self._session_id,
                    )
                )
                return {"status": "success", "memory_id": res.memory_id}
            if name == "memory_delete":
                mid = args.get("memory_id", "")
                if not mid:
                    return {"error": "memory_id is required"}
                self._runner.run(self._backend.delete(mid))
                return {"status": "success"}
            if name == "memory_list":
                items = self._runner.run(
                    self._backend.list(
                        user_id=self._user_id, agent_id=self._agent_id,
                        limit=int(args.get("limit") or 20),
                    )
                )
                return {
                    "status": "success",
                    "count": len(items),
                    "memories": [{"memory_id": m.memory_id, "content": m.content} for m in items],
                }
            return {"error": f"Unknown tool: {name}"}
        except Exception as exc:
            logger.error("[%s] tool %s failed: %s", self.name, name, exc)
            return {"error": str(exc)}

    # ---- 关闭 ------------------------------------------------------------
    def shutdown(self) -> None:
        with self._lock:
            self._flush_session_buffer(None)
            self._wait_inflight()
            if self._executor is not None:
                try:
                    self._executor.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass
                self._executor = None
            if self._runner is not None:
                try:
                    self._runner.close()
                except Exception:
                    pass
                self._runner = None
            self._backend = None
            self._initialized = False

    def _wait_inflight(self, timeout_sec: float = 10.0) -> None:
        if not self._inflight:
            return
        pending = [f for f in self._inflight if not f.done()]
        if not pending:
            return
        per_each = max(0.1, timeout_sec / len(pending))
        for f in pending:
            try:
                f.result(timeout=per_each)
            except Exception:
                pass
        self._inflight.clear()
