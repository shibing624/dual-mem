# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Public async facade of the dual-mem layered-memory SDK. MemoryClient wires
configured providers/stores together and exposes add/search/get/list/update/delete/digest,
returning strongly typed dataclass models (sdk_models). Multi-turn messages are formatted
into natural dialogue text for L1_RAW + extract; embed sees the user-only concatenation.
A per-user asyncio.Lock serializes concurrent add() for the same (app_id, user_id) so the
fast-write -> reconcile pipeline can never race on the same evolution chain.

dual-mem requires both an LLM API key and an embedding API key; missing credentials raise
on construction (no silent embedding-only fallback). When you need to inject your own LLM
client (tests, custom backends), pass it via the ``llm=`` constructor kwarg.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from typing import Any

from dual_mem.config import Settings
from dual_mem.isolation import build_filter
from dual_mem.locks import LockRegistry
from dual_mem.registry import _UNSET, ComponentFactory
from dual_mem.retrieval.reader import Reader
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
from dual_mem.system2.cross_domain_sweeper import CrossDomainSweeper
from dual_mem.system2.system2_writer import System2Writer
from dual_mem.types import MemoryStatus
from dual_mem.writer.memory_writer import MemoryWriter

logger = logging.getLogger("dual_mem.client")


class MissingCredentialsError(RuntimeError):
    """Raised when MemoryClient is constructed without the required LLM / embedding keys."""


class MemoryClient:
    """High-level async entry point for dual-mem.

    Picks ``system1`` (MemoryWriter) or ``dual`` (System2Writer) from ``Settings.mode``.

    Constructor kwargs (see also ``Settings`` / ``~/.dual_mem/config.yaml``):

    - ``mode``: ``"system1"`` (default, L0–L4 fast-write) or ``"dual"`` (+ async System2 L6/L7).
    - ``storage_dir``: on-disk root for Chroma / Kuzu / SQLite; default ``./.dual_mem_data``.
    - ``settings``: explicit ``Settings`` instance; overrides YAML and env.
    - ``embed`` / ``llm``: inject custom clients; skips the corresponding api_key check.

    Tenant scope on ``add`` / ``search``:

    - ``app_id`` (optional): defaults to ``settings.default_app_id`` (``"default"``).
    - ``user_id`` (required): end-user id; reads and writes must share the same scope.
    - ``app_ids`` (optional on search): defaults to ``[default_app_id]``.
    - ``agent_id`` / ``session_id`` (optional): finer isolation within one user.

    Lifecycle: reuse one client per process (FastAPI lifespan / agent runtime). Call
    ``await aclose()`` on shutdown when ``mode="dual"`` and ``system2_trigger_mode="scheduled"``;
    optional otherwise. Do **not** call ``aclose()`` after every add/search.
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
        overrides = {}
        if storage_dir is not None:
            overrides["storage_dir"] = storage_dir
        if mode is not None:
            overrides["mode"] = mode
        if settings is None:
            settings = Settings(**overrides)
        elif overrides:
            settings = settings.model_copy(update=overrides)
        self.settings = settings
        self.mode = mode or settings.mode

        self.factory = ComponentFactory(
            settings=settings,
            embed=embed,
            llm=llm if llm is not None else _UNSET,
        )

        self._validate_credentials(injected_embed=embed is not None, injected_llm=llm is not None)
        self._build_writer()
        self.reader = Reader(factory=self.factory)
        self._write_locks = LockRegistry()

    @classmethod
    def from_config(
        cls,
        config_dict: dict[str, Any],
        *,
        mode: str | None = None,
        embed=None,
        llm=None,
    ) -> MemoryClient:
        """Create a client from a mem0/Hy-style config dict (see ``Settings.from_dict``).

        Usage::

            client = MemoryClient.from_config({
                "mode": "dual",
                "default_app_id": "my_app",
                "llm": {"model": "gpt-4o-mini", "api_key": "sk-..."},
                "embedder": {"model": "text-embedding-3-small", "api_key": "sk-..."},
            })
        """
        settings = Settings.from_dict(config_dict)
        if mode is not None:
            settings = settings.model_copy(update={"mode": mode})
        return cls(settings=settings, embed=embed, llm=llm)

    @classmethod
    async def acreate(
        cls,
        *,
        settings: Settings | None = None,
        storage_dir: str | None = None,
        mode: str | None = None,
        embed=None,
        llm=None,
    ) -> "MemoryClient":
        """Async factory mirror of ``MemoryClient(...)``; kept for API symmetry with prior versions."""
        return cls(
            settings=settings,
            storage_dir=storage_dir,
            mode=mode,
            embed=embed,
            llm=llm,
        )

    def _validate_credentials(self, *, injected_embed: bool, injected_llm: bool) -> None:
        """Fail fast when LLM or embedding credentials are missing.

        Injected ``embed`` / ``llm`` clients (tests, custom backends) bypass the api_key check
        — the caller is responsible for those. Raises ``MissingCredentialsError`` otherwise.
        """
        missing: list[str] = []
        if not injected_llm and not self.settings.llm_api_key:
            missing.append("llm_api_key (set DUAL_MEM_LLM_API_KEY or pass settings.llm_api_key)")
        if not injected_embed and not self.settings.embed_api_key:
            missing.append("embed_api_key (set DUAL_MEM_EMBED_API_KEY or pass settings.embed_api_key)")
        if missing:
            raise MissingCredentialsError(
                "dual-mem requires both LLM and embedding API keys. Missing: "
                + "; ".join(missing)
                + ". For tests / custom backends, pass embed=... and llm=... to MemoryClient(...)."
            )

    def _build_writer(self) -> None:
        """Instantiate the writer matching the configured mode (system1 or dual)."""
        if self.settings.mode == "dual":
            self.writer = System2Writer(factory=self.factory)
        else:
            self.writer = MemoryWriter(factory=self.factory)

    def _resolve_app_id(self, app_id: str | None) -> str:
        """Return explicit ``app_id`` or ``settings.default_app_id``."""
        return app_id if app_id is not None else self.settings.default_app_id

    def _user_write_lock(self, app_id: str, user_id: str) -> asyncio.Lock:
        """Return the per-user write lock, creating it lazily on first access."""
        return self._write_locks.get(f"{app_id}::{user_id}")

    def _user_write_lock_ctx(self, app_id: str, user_id: str):
        """Per-user write lock, or a no-op context when ``write_serialize_per_user`` is off.

        Disabling the lock lets concurrent add() for the same user overlap (batch ingest with
        deferred reconcile); the vector store is internally thread-safe so this is safe.
        """
        if self.settings.write_serialize_per_user:
            return self._user_write_lock(app_id, user_id)
        return contextlib.nullcontext()

    async def add(
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
        """Write one memory and run the cognition pipeline (Gate → Extract → …).

        Pass either ``content`` (single blob) or ``messages`` (multi-turn chat) — same API.

        ``app_id`` defaults to ``settings.default_app_id`` when omitted.
        ``user_id`` is required. With ``messages``,
        only ``role=='user'`` turns drive Gate vector novelty (max across turns); L1_RAW
        and the extractor still see the full dialogue; the last assistant turn feeds Gate LLM context.

        Each add costs ~2 LLM calls (Gate + Extract). For agent apps, you may batch
        ``messages`` at session end to reduce cost; ``add(content=...)`` per turn is
        also supported when low-latency persistence is required — see docs/skills tradeoff.
        """
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        resolved_app_id = self._resolve_app_id(app_id)

        user_queries: list[str] = []
        agent_context: str | None = None
        if messages:
            normalized = _normalize_messages(messages)
            cpt = self.settings.chars_per_token
            threshold_chars = int(
                self.settings.llm_context_window
                * self.settings.extract_history_context_ratio
                * cpt
            )
            normalized = _shape_history(
                normalized,
                threshold_chars=threshold_chars,
                assistant_max_chars=int(self.settings.extract_assistant_max_tokens * cpt),
            )
            user_queries = [m.content for m in normalized if m.role == "user" and m.content.strip()]
            content = _format_dialogue(normalized)
            agent_context = _last_assistant_context(normalized)

        logger.info(
            "add app=%s user=%s mode=%s len=%d turns=%d",
            resolved_app_id, user_id, self.mode, len(content), len(user_queries),
        )

        async with self._user_write_lock_ctx(resolved_app_id, user_id):
            result = await self.writer.write(
                content=content,
                app_id=resolved_app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                request_id=request_id,
                memory_at=memory_at,
                user_queries=user_queries or None,
                agent_context=agent_context,
            )
        return WriteResult(
            success=True,
            memory_id=result.memory_id,
            request_id=request_id,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            gate_passed=result.gate_passed,
            gate_score=result.gate_score,
            extracted_count=len(result.extra_node_ids),
            extra_node_ids=list(result.extra_node_ids),
            is_ephemeral=result.is_ephemeral,
        )

    async def search(
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
        """Semantic search; returns profile / proactive / normal groups.

        读路径只用 embedding + 混合检索，无 LLM，可在 agent 每轮生成前安全调用。
        Read path uses embedding + hybrid retrieval — no LLM; safe to call every user turn.

        Args:
            query: 查询文本 / the query string to recall against.
            app_ids: 应用隔离域，省略时用 ``[settings.default_app_id]`` /
                app isolation scope; defaults to ``[settings.default_app_id]``.
            user_id: 用户隔离域，必须与 ``add`` 时一致 / user scope, must match ``add``.
            agent_ids: 智能体隔离域（多 agent 共享同一 user 时用）/
                agent scope (when several agents share one user).
            session_ids: 限定在某些会话内检索 / restrict recall to specific sessions.
            limit: normal 路（L2/L5/L3/L1 事实类）返回上限 /
                cap on the normal route (episodic facts L2/L5/L3/L1).
            min_score: normal 路最低融合分，低于此分丢弃 / min fused score for the normal route.
            profile_limit: profile 路（L0/L4/L6 画像类）上限，``-1`` = 不限 /
                cap on the profile route (identity/schema), ``-1`` = unlimited.
            profile_min_score: profile 路最低分 / min score for the profile route.
            intention_limit: proactive 路（L7 意图）返回条数，``0`` = 关闭。意图是“用户接下来
                想做什么”的主动提醒，仅在需要主动推荐时才开；事实问答场景保持 0，以免意图噪声
                挤占事实召回。 / proactive route (L7 intentions) size, ``0`` = OFF. Intentions
                are forward-looking "what the user plans to do" nudges — only enable for
                proactive recommendation; keep 0 for factual QA so intention noise does not
                crowd out facts.
            created_after: 只返回该 Unix 秒之后创建的记忆（时间窗过滤）/
                only memories created after this Unix-second cutoff.
            request_id: 日志/链路追踪 id，省略自动生成 / trace id for logs; auto-generated.
            debug: ``True`` 时填充 ``SearchResult.read_result`` 的逐阶段 trace /
                fill ``SearchResult.read_result`` with per-stage trace metadata.
        """
        request_id = request_id or str(uuid.uuid4())
        start = time.perf_counter()
        resolved_app_ids = app_ids if app_ids is not None else [self.settings.default_app_id]
        logger.info(
            "search app=%s user=%s query=%r limit=%d debug=%s",
            (resolved_app_ids[0] if resolved_app_ids else ""), user_id, query, limit, debug,
        )
        memories, trace = await self.reader.search(
            query=query,
            app_ids=resolved_app_ids,
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
            collect_trace=debug,
        )

        # In per_write mode, drain reconsolidation tasks the read hook just enqueued so
        # users do not have to wait for the next write to see the reactivation effects.
        # The hook runs as a fire-and-forget task inside the reader, so await it FIRST —
        # otherwise the drain races ahead of its own enqueue and finds an empty queue.
        if (
            isinstance(self.writer, System2Writer)
            and self.settings.system2_trigger_mode == "per_write"
        ):
            enqueue_task = self.reader.last_reconsolidation_task
            if enqueue_task is not None:
                try:
                    await enqueue_task
                except Exception:
                    pass  # enqueue failures are logged by the hook's done-callback
            task = asyncio.create_task(self.writer._digest_reconsolidation_pending())
            task.add_done_callback(_swallow_task_exception)

        return SearchResult(
            success=True,
            request_id=request_id,
            memories=memories,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            read_result=trace,
        )

    async def get(self, memory_id: str) -> MemoryItem | None:
        """Fetch a single memory by id, or None if it does not exist."""
        node = self.factory.vector.get(memory_id)
        if node is None:
            return None
        return Reader.memory_node_to_item(node)

    async def list(
        self, *, app_id: str | None = None, user_id: str, agent_id: str = "", limit: int = 100
    ) -> list[MemoryItem]:
        """List ACTIVE memories under the given app/user (optionally agent) scope."""
        resolved_app_id = self._resolve_app_id(app_id)
        where = build_filter(
            app_ids=[resolved_app_id],
            user_id=user_id,
            agent_ids=[agent_id],
            statuses=[MemoryStatus.ACTIVE],
        )
        nodes = self.factory.vector.get_many(where, limit=limit)
        return [Reader.memory_node_to_item(node) for node in nodes]

    async def update(self, memory_id: str, content: str) -> UpdateResult:
        """Replace a memory's content and re-embed it, logging the change to history."""
        node = self.factory.vector.get(memory_id)
        if node is None:
            return UpdateResult(success=False, error_code=404)
        old_meta = node.to_metadata()
        node.content = content
        node.embedding = await self.factory.embed.embed(content)
        node.gmt_modified = int(time.time())
        self.factory.vector.upsert([node])
        self.factory.history.append(
            event="UPDATE",
            node_id=node.node_id,
            user_id=node.user_id,
            old=old_meta,
            new=node.to_metadata(),
        )
        return UpdateResult(success=True, memory_id=memory_id)

    async def delete(self, memory_id: str) -> DeleteResult:
        """Physically remove a memory and append a DELETE history record."""
        node = self.factory.vector.get(memory_id)
        if node is None:
            return DeleteResult(success=False, error_code=404)
        self.factory.vector.delete([memory_id])
        self.factory.history.append(
            event="DELETE",
            node_id=memory_id,
            user_id=node.user_id,
            old=node.to_metadata(),
            new=None,
        )
        return DeleteResult(success=True)

    async def delete_bulk(
        self,
        *,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        confirm: bool = False,
    ) -> DeleteBulkResult:
        """Delete every memory in a scope; requires confirm=True as a safety guard."""
        if confirm is not True:
            return DeleteBulkResult(success=False, error_code=400)
        resolved_app_id = self._resolve_app_id(app_id)
        where: dict = {"app_id": {"$in": [resolved_app_id]}}
        if user_id is not None:
            where["user_id"] = user_id
        if agent_id is not None:
            where["agent_id"] = agent_id
        nodes = self.factory.vector.get_many(where)
        node_ids = [node.node_id for node in nodes]
        self.factory.vector.delete(node_ids)
        graph = self.factory.graph
        if graph is not None:
            graph.delete_scope(
                app_id=resolved_app_id,
                user_id=user_id,
                agent_id=agent_id,
            )
        if self.settings.persist_history:
            for node in nodes:
                self.factory.history.append(
                    event="DELETE",
                    node_id=node.node_id,
                    user_id=node.user_id,
                    old=node.to_metadata(),
                    new=None,
                )
        return DeleteBulkResult(success=True, deleted=len(node_ids))

    async def list_scopes(
        self,
        *,
        app_id: str | None = None,
        limit: int = 5000,
    ) -> list[ScopeSummary]:
        """List distinct memory scopes (app_id, user_id, agent_id) present in storage."""
        where: dict = {}
        if app_id is not None:
            where["app_id"] = app_id
        nodes = self.factory.vector.get_many(where, limit=limit)
        counts: dict[tuple[str, str, str], int] = {}
        for node in nodes:
            key = (node.app_id, node.user_id, node.agent_id or "")
            counts[key] = counts.get(key, 0) + 1
        return [
            ScopeSummary(
                app_id=key[0],
                user_id=key[1],
                agent_id=key[2],
                memory_count=count,
            )
            for key, count in sorted(counts.items())
        ]

    async def digest(self) -> DigestResult:
        """Drain every pending System2 task: reconcile chains, run S2 agent, sweeper."""
        if not isinstance(self.writer, System2Writer):
            return DigestResult(success=True, processed=0)
        processed = await self.writer._digest_pending()
        timing = dict(self.writer.last_digest_stats)
        cores = 0
        if self.settings.cross_domain_enable:
            sweeper = CrossDomainSweeper(factory=self.factory)
            for app_id, user_id in self.writer.processed_pairs:
                result = await sweeper.run(app_id=app_id, user_id=user_id)
                if result.get("cores"):
                    cores += int(result["cores"])
        if self.settings.purge_done_queues:
            self.factory.cache.purge_done_queues()
        return DigestResult(
            success=True, processed=processed, cores_created=cores, timing=timing
        )

    async def aclose(self) -> None:
        """Release dual-mode background resources. Idempotent.

        Cancels the scheduled System2 loop when ``system2_trigger_mode="scheduled"`` and
        releases the embedded Kuzu graph lock. No-op extras for ``system1``. Does **not**
        await in-flight ``per_write`` digest tasks. Call once at application shutdown.
        """
        if isinstance(self.writer, System2Writer):
            await self.writer.aclose()
        self.factory.close()


def _swallow_task_exception(task: asyncio.Task) -> None:
    """Drop any exception raised in a fire-and-forget background task."""
    if task.cancelled():
        return
    task.exception()


def _normalize_messages(messages: list) -> list[ChatMessage]:
    """Coerce the user's input (mixed dict / ChatMessage) into a clean ChatMessage list."""
    out: list[ChatMessage] = []
    for msg in messages:
        if isinstance(msg, ChatMessage):
            role = (msg.role or "user").lower()
            text = (msg.content or "").strip()
        elif isinstance(msg, dict):
            role = str(msg.get("role") or "user").lower()
            text = str(msg.get("content") or "").strip()
        else:
            continue
        if not text:
            continue
        out.append(ChatMessage(role=role, content=text))
    return out


def _shape_history(
    messages: list[ChatMessage],
    *,
    threshold_chars: int,
    assistant_max_chars: int,
) -> list[ChatMessage]:
    """Shape multi-turn input before extract WITHOUT dropping any turn.

    Truncation only kicks in when the whole dialogue is large: if the total content length is
    within ``threshold_chars`` (derived from the model context window), the dialogue passes
    through untouched — short batches keep their assistant turns in full. Above the threshold,
    non-user turns (assistant/system — the model's own words) are truncated to
    ``assistant_max_chars``; user turns (the real memory signal) are always preserved in full.

    ``threshold_chars<=0`` or ``assistant_max_chars<=0`` disables shaping entirely.
    """
    if threshold_chars <= 0 or assistant_max_chars <= 0:
        return messages
    total_chars = sum(len(m.content) for m in messages)
    if total_chars <= threshold_chars:
        return messages
    shaped: list[ChatMessage] = []
    for msg in messages:
        if msg.role == "user" or len(msg.content) <= assistant_max_chars:
            shaped.append(msg)
        else:
            shaped.append(
                ChatMessage(role=msg.role, content=msg.content[:assistant_max_chars] + "…")
            )
    return shaped


def _format_dialogue(messages: list[ChatMessage]) -> str:
    """Render normalized messages into a natural-language dialogue chunk for L1_RAW + extract."""
    parts: list[str] = []
    for msg in messages:
        if msg.role == "user":
            label = "[user]"
        elif msg.role == "assistant":
            label = "[assistant]"
        elif msg.role == "system":
            label = "[system]"
        else:
            label = f"[{msg.role}]"
        parts.append(f"{label}: {msg.content}")
    return "\n".join(parts)


def _last_assistant_context(messages: list[ChatMessage]) -> str | None:
    """Return the last non-empty assistant turn for Gate context scoring."""
    for msg in reversed(messages):
        if msg.role == "assistant" and msg.content.strip():
            return msg.content.strip()
    return None
