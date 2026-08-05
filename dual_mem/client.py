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

    Lifecycle: reuse one client per process (FastAPI lifespan / agent runtime) and call
    ``await aclose()`` once at shutdown. Do **not** close it after every add/search.
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
        self._coding_writer = None
        if self.settings.coding_enabled and self.factory.llm is not None:
            self._init_coding_writer()

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

    def _init_coding_writer(self) -> None:
        """Create the coding memory subsystem (separate store + extractor + writer)."""
        import os
        from dual_mem.coding.store import CodingMemoryStore
        from dual_mem.coding.extractor import CodingMemoryExtractor
        from dual_mem.coding.reconciler import CodingMemoryReconciler
        from dual_mem.coding.writer import CodingWriter

        db_path = self.settings.coding_db_path or os.path.join(
            self.settings.storage_dir, "coding_memory.db"
        )
        store = CodingMemoryStore(
            db_path=db_path,
            vector=self.factory.vector,
            embed=self.factory.embed,
        )
        extractor = CodingMemoryExtractor(
            llm=self.factory.llm,
            tool_result_max_bytes=self.settings.coding_tool_result_max_bytes,
        )
        self._coding_writer = CodingWriter(
            store=store,
            extractor=extractor,
            reconciler=CodingMemoryReconciler(),
            llm=self.factory.llm,
        )

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
        """Write one memory and run the cognition pipeline (Extract -> commit -> ...).

        Pass either ``content`` (single blob) or ``messages`` (multi-turn chat) — same API.

        ``app_id`` defaults to ``settings.default_app_id`` when omitted.
        ``user_id`` is required. With ``messages``, L1_RAW and the extractor see the shaped
        full dialogue after host-injected system turns are removed.

        Each add performs one extraction call; long content may also trigger the optional
        summarizer. Batch ``messages`` at session end to reduce cost when appropriate.
        """
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        resolved_app_id = self._resolve_app_id(app_id)

        user_turn_count = 0
        if messages:
            # Coding path: check for tool messages before chat normalization
            if self._coding_writer:
                raw_dicts = [
                    m if isinstance(m, dict) else {"role": m.role, "content": m.content}
                    for m in messages
                ]
                from dual_mem.coding.preproc import has_any_tool_message, strip_tool_messages
                if has_any_tool_message(raw_dicts):
                    try:
                        coding_result = await self._coding_writer.write(
                            raw_dicts, user_id=user_id, agent_id=agent_id,
                            session_id=session_id, app_id=resolved_app_id,
                        )
                    except Exception as exc:
                        logger.warning("coding write failed: %s; falling back to chat", exc)
                        coding_result = None
                    if coding_result is not None:
                        mids = coding_result.get("memory_ids") or []
                        return WriteResult(
                            success=True,
                            memory_id=mids[0] if mids else "",
                            request_id=request_id,
                            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
                        )
                    # Not coding → strip tool messages so chat path sees clean dialogue
                    messages = strip_tool_messages(raw_dicts)
            normalized = _normalize_messages(messages)
            # Host-injected role=system turns (e.g. "You are a helpful assistant") are not
            # user memory — drop before L1/extract. Extract LLM's own EXTRACT_* template is
            # separate instruction, not part of this dialogue.
            normalized = [m for m in normalized if m.role != "system"]
            cpt = self.settings.chars_per_token
            threshold_chars = int(
                self.settings.llm_context_window
                * self.settings.extract_dialogue_context_ratio
                * cpt
            )
            normalized = _shape_history(
                normalized,
                threshold_chars=threshold_chars,
                assistant_max_chars=int(self.settings.extract_assistant_max_tokens * cpt),
            )
            user_turn_count = sum(
                1 for m in normalized if m.role == "user" and m.content.strip()
            )
            content = _format_dialogue(normalized)

        logger.info(
            "add app=%s user=%s mode=%s len=%d turns=%d",
            resolved_app_id, user_id, self.mode, len(content), user_turn_count,
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
            )
        return WriteResult(
            success=True,
            memory_id=result.memory_id,
            request_id=request_id,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            commit_passed=result.commit_passed,
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
        include_derived: bool = True,
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
            limit: normal 路（L2/L3/L4）返回上限 /
                cap on the normal route (L2/L3/L4 memories).
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
            include_derived: 是否返回 System2 派生的 L6/L7；事实型评测应显式设为
                ``False``，偏好与泛化任务保持 ``True``。
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
            include_derived=include_derived,
            created_after=created_after,
            request_id=request_id,
            collect_trace=debug,
        )

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
        """Explicitly drain pending reconcile work and run System2 cognition."""
        if not isinstance(self.writer, System2Writer):
            return DigestResult(success=True, processed=0)
        processed = await self.writer.digest_pending()
        timing = dict(self.writer.last_digest_stats)
        if self.settings.purge_done_queues:
            self.factory.cache.purge_done_queues()
        return DigestResult(success=True, processed=processed, timing=timing)

    async def search_coding(
        self, *, query: str, user_id: str, agent_id: str = "default_agent",
        app_id: str | None = None, top_k: int = 10,
    ) -> list[dict]:
        """Search coding memories (tool-use / engineering memories)."""
        if not self._coding_writer:
            return []
        resolved = self._resolve_app_id(app_id)
        return await self._coding_writer.search(
            query=query, user_id=user_id, agent_id=agent_id,
            app_id=resolved, top_k=top_k,
        )

    async def aclose(self) -> None:
        """Release storage resources. Idempotent; call once at application shutdown."""
        if self._coding_writer is not None:
            self._coding_writer.store.close()
        self.factory.close()


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
    assistant turns are truncated to ``assistant_max_chars``; user turns (the primary memory
    signal) are always preserved in full. Callers should omit ``role=system`` dialogue turns
    (filtered earlier in ``MemoryClient.add``); they are not memory content.

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
