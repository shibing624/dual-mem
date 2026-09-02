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
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from dual_mem.client import MemoryClient, strip_memory_injection
from dual_mem.config import Settings, resolve_app_id
from dual_mem.isolation import build_filter
from dual_mem.retrieval.reader import Reader
from dual_mem.sdk_models import MemoryItem
from dual_mem.types import Layer, MemoryStatus

logger = logging.getLogger("dual_mem.integrations")

MEMORY_TOOLS_GUIDE = (
    "<memory-tools-guide>\n"
    "当上方记忆不足以回答时，可调用 memory_search 检索结构化记忆；\n"
    "需要核对原话、时间或来源时，调用 conversation_search，"
    "或按注入块里的来源 id 使用 memory 的 get。\n"
    "每轮对话中，memory_search 与 conversation_search 合计最多调用 3 次。\n"
    "若 3 次后仍无结果，说明该信息不在记忆中，请直接基于已有信息回复，不要继续搜索。\n"
    "</memory-tools-guide>"
)

_SYSTEM_INTRO = (
    "You have access to dual-mem — a persistent layered memory that "
    "remembers user preferences, facts, and context across sessions. "
    "Relevant memories are automatically provided before each response."
)


@dataclass
class RenderedMemoryContext:
    """Split injection: query-independent profile vs this-turn facts."""

    stable_block: str
    dynamic_block: str
    total_chars: int


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
        stamps = mem.merged_timestamps or []
        if len(stamps) > 1:
            start = _fmt_time(min(stamps))[:10]
            end = _fmt_time(max(stamps))[:10]
            if start and end:
                entry = f"{entry}\n  (持续: {start} → {end})"
        if mem.source_node_id:
            entry = f"{entry}\n  (来源: L1 {mem.source_node_id[:8]})"
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


def format_profile_block(
    items: Iterable[MemoryItem],
    *,
    max_chars: int = 2000,
) -> str:
    """Render query-independent L0 profile items, sorted by id for byte-stable output."""
    rows = [
        (mem.memory_id, (mem.content or "").strip())
        for mem in items
        if (mem.content or "").strip()
    ]
    if not rows:
        return ""
    rows.sort(key=lambda row: row[0])
    out: list[str] = []
    running = 0
    for _, content in rows:
        entry = f"- {content}"
        if running + len(entry) > max_chars:
            break
        out.append(entry)
        running += len(entry) + 1
    if not out:
        return ""
    return (
        "<user-profile>\n"
        "Stable user profile (does not change with the current query):\n"
        + "\n".join(out)
        + "\n</user-profile>"
    )


def format_topic_catalog(
    tags: Iterable[str],
    schemas: Iterable[str] | None = None,
    *,
    max_chars: int = 800,
) -> str:
    """Query-independent topic / schema directory for the stable system tail."""
    items = sorted({
        str(item).strip()
        for item in (*tags, *(schemas or ()))
        if item and str(item).strip()
    })
    if not items:
        return ""
    out: list[str] = []
    running = 0
    for item in items:
        entry = f"- {item}"
        if running + len(entry) > max_chars:
            break
        out.append(entry)
        running += len(entry) + 1
    if not out:
        return ""
    return (
        "<topic-catalog>\n"
        "Known memory topics (does not change with the current query). "
        "Use memory_search for facts, conversation_search for original quotes:\n"
        + "\n".join(out)
        + "\n</topic-catalog>"
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

    async def add_raw(
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
        return await self.client.add_raw(
            content=content,
            messages=messages,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            memory_at=memory_at,
        )

    async def distill(
        self,
        *,
        user_id: str,
        source_node_ids: list[str],
        app_id: Optional[str] = None,
        content: str = "",
        messages: Optional[list] = None,
        agent_id: str = "",
        session_id: str = "",
        memory_at: Optional[int] = None,
    ) -> Any:
        return await self.client.distill(
            content=content,
            messages=messages,
            user_id=user_id,
            source_node_ids=source_node_ids,
            app_id=app_id,
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
        profile_limit: Optional[int] = None,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        include_derived: Optional[bool] = None,
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
            include_derived=include_derived,
            created_after=created_after,
        )

    async def search_conversation(
        self,
        *,
        query: str,
        user_id: str,
        app_ids: Optional[list[str]] = None,
        agent_ids: Optional[list[str]] = None,
        session_ids: Optional[list[str]] = None,
        limit: int = 10,
        min_score: float = 0.0,
        created_after: Optional[int] = None,
    ) -> Any:
        return await self.client.search_conversation(
            query=query,
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            limit=limit,
            min_score=min_score,
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

    async def load_profile_block(
        self,
        *,
        user_id: str,
        app_id: Optional[str] = None,
        limit: int = 50,
        max_chars: int = 2000,
    ) -> str:
        """List ACTIVE L0 profile nodes and a query-independent topic catalog."""
        resolved = resolve_app_id(self.client.settings, app_id)
        where = build_filter(
            app_ids=[resolved],
            user_id=user_id,
            layers=[Layer.L0_BASIC_INFO],
            statuses=[MemoryStatus.ACTIVE],
        )
        nodes = await asyncio.to_thread(self.client.factory.vector.get_many, where, limit)
        fact_where = build_filter(
            app_ids=[resolved],
            user_id=user_id,
            layers=[Layer.L2_FACT, Layer.L4_IDENTITY],
            statuses=[MemoryStatus.ACTIVE],
        )
        fact_nodes = await asyncio.to_thread(
            self.client.factory.vector.get_many, fact_where, 200,
        )
        tags = [tag for node in fact_nodes for tag in node.tags]
        schemas: list[str] = []
        graph = self.client.factory.graph
        if graph is not None:
            for schema in graph.list_by_layer(
                layer=Layer.L6_SCHEMA.value,
                user_id=user_id,
                app_ids=[resolved],
                limit=40,
            ):
                title = (schema.content or "").strip().splitlines()[0][:40]
                if title:
                    schemas.append(title)
        parts = [
            format_profile_block(
                [Reader.memory_node_to_item(node) for node in nodes],
                max_chars=max_chars,
            ),
            format_topic_catalog(tags, schemas),
        ]
        return "\n\n".join(part for part in parts if part)

    async def render_split_context(
        self,
        *,
        query: str,
        user_id: str,
        app_ids: Optional[list[str]] = None,
        agent_ids: Optional[list[str]] = None,
        limit: int = 10,
        max_chars: int = 2000,
        **kwargs: Any,
    ) -> RenderedMemoryContext:
        """Stable L0 dump + this-turn normal-route facts."""
        app_id = app_ids[0] if app_ids else None
        stable = await self.load_profile_block(
            user_id=user_id, app_id=app_id, max_chars=max_chars,
        )
        result = await self.search(
            query=query, user_id=user_id, app_ids=app_ids,
            agent_ids=agent_ids, limit=limit, **kwargs,
        )
        dynamic = format_memories_for_prompt(result.memories.normal, max_chars=max_chars)
        return RenderedMemoryContext(
            stable_block=stable,
            dynamic_block=dynamic,
            total_chars=len(stable) + len(dynamic),
        )

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
    _max_prefetch_timeout_ms: int = 3000
    _max_search_calls_per_turn: int = 3
    _search_calls_this_turn: int = 0
    _stable_profile: str = ""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        storage_dir: Optional[str] = None,
        mode: Optional[str] = None,
        write_turn_window: int = 5,
        idle_timeout_sec: float = 30.0,
        max_prefetch_chars: int = 2000,
        max_prefetch_timeout_ms: int = 3000,
        max_search_calls_per_turn: int = 3,
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
        self._idle_timeout_sec = float(idle_timeout_sec)
        self._idle_timers: dict[str, threading.Timer] = {}
        self._l1_ids: dict[str, list[str]] = {}
        self._max_prefetch_chars = max(200, int(max_prefetch_chars))
        self._max_prefetch_timeout_ms = max(1, int(max_prefetch_timeout_ms))
        self._max_search_calls_per_turn = max(1, int(max_search_calls_per_turn))
        self._search_calls_this_turn = 0
        self._stable_profile = ""
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
            if self._user_id:
                self._stable_profile = self._runner.run(
                    self._backend.load_profile_block(user_id=self._user_id)
                )
            self._initialized = True

    # ---- 上下文注入（被动） ----------------------------------------------
    def prefetch(self, query: str, **kwargs: Any) -> str:
        self._search_calls_this_turn = 0
        if not self._backend or not self._runner or not query:
            return ""
        q = query.strip()
        if not q:
            return ""

        async def _timed_search():
            return await asyncio.wait_for(
                self._backend.search(
                    query=q, user_id=self._user_id,
                    agent_ids=[self._agent_id], limit=10,
                ),
                timeout=self._max_prefetch_timeout_ms / 1000.0,
            )

        try:
            result = self._runner.run(_timed_search())
            items = result.memories.normal
            if not items:
                return ""
            return format_memories_for_prompt(items, max_chars=self._max_prefetch_chars)
        except Exception as exc:  # prefetch must never block the host session
            logger.warning("[%s] prefetch failed: %s", self.name, exc)
            return ""

    def queue_prefetch(self, query: str, **kwargs: Any) -> None:
        return

    def system_prompt_block(self) -> str:
        parts = [_SYSTEM_INTRO, MEMORY_TOOLS_GUIDE]
        if self._stable_profile:
            parts.append(self._stable_profile)
        return "\n\n".join(parts)

    # ---- 写入节流（sync_turn） -------------------------------------------
    def sync_turn(self, user_message: str, assistant_response: str, **kwargs: Any) -> None:
        if not self._backend or not self._runner or not user_message:
            return
        cleaned = strip_memory_injection(user_message)
        if not cleaned:
            return
        session_id = kwargs.get("session_id") or self._session_id or "default_session"
        turn = [
            {"role": "user", "content": cleaned},
            {"role": "assistant", "content": assistant_response or ""},
        ]
        try:
            raw = self._runner.run(
                self._backend.add_raw(
                    messages=turn, user_id=self._user_id,
                    agent_id=self._agent_id, session_id=session_id,
                )
            )
        except Exception as exc:
            logger.warning("[%s] add_raw failed: %s", self.name, exc)
            return
        with self._buffer_lock:
            buf = self._turn_buffer.setdefault(session_id, [])
            buf.extend(turn)
            if raw.memory_id:
                self._l1_ids.setdefault(session_id, []).append(raw.memory_id)
            turns = sum(1 for m in buf if m["role"] == "user")
            ready = turns >= self._write_turn_window
        self._arm_idle(session_id)
        if ready:
            self._idle_flush(session_id)

    def _arm_idle(self, session_id: str) -> None:
        self._cancel_idle(session_id)
        if self._idle_timeout_sec <= 0:
            return
        timer = threading.Timer(self._idle_timeout_sec, self._idle_flush, args=(session_id,))
        timer.daemon = True
        timer.start()
        self._idle_timers[session_id] = timer

    def _cancel_idle(self, session_id: str) -> None:
        timer = self._idle_timers.pop(session_id, None)
        if timer is not None:
            timer.cancel()

    def _idle_flush(self, session_id: str) -> None:
        self._cancel_idle(session_id)
        with self._buffer_lock:
            msgs = self._turn_buffer.get(session_id) or []
            l1_ids = list(self._l1_ids.get(session_id) or [])
            self._turn_buffer[session_id] = []
            self._l1_ids[session_id] = []
        if not msgs or not self._backend or not self._runner:
            return
        try:
            self._runner.run(
                self._backend.distill(
                    messages=msgs, user_id=self._user_id,
                    source_node_ids=l1_ids,
                    agent_id=self._agent_id, session_id=session_id,
                )
            )
        except Exception as exc:
            logger.warning("[%s] distill failed: %s", self.name, exc)

    def _flush_session_buffer(self, session_id: Optional[str] = None) -> None:
        if session_id is None:
            with self._buffer_lock:
                sids = [sid for sid, msgs in self._turn_buffer.items() if msgs]
        else:
            sids = [session_id]
        for sid in sids:
            self._idle_flush(sid)

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
                "name": "conversation_search",
                "description": (
                    "Search original conversation text (L1_RAW) to verify quotes, "
                    "time, or source. Do not use for default fact recall."
                ),
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
            if name in ("memory_search", "conversation_search"):
                if self._search_calls_this_turn >= self._max_search_calls_per_turn:
                    return {
                        "status": "limit_reached",
                        "hint": "已达本轮检索上限，请基于已有记忆回答",
                    }
                self._search_calls_this_turn += 1
                search = (
                    self._backend.search_conversation
                    if name == "conversation_search"
                    else self._backend.search
                )
                result = self._runner.run(
                    search(
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
            for sid in list(self._idle_timers):
                self._cancel_idle(sid)
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
