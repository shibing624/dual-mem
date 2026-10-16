# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Public async facade of the dual-mem layered-memory SDK. MemoryClient wires
configured providers/stores together and exposes add/search/get/list/update/delete/digest,
returning strongly typed dataclass models (sdk_models). Multi-turn messages are formatted
into natural dialogue text for L0_RAW + extract; embed sees the user-only concatenation.
A per-user asyncio.Lock serializes concurrent add() for the same (app_id, user_id) so
version appends cannot race on the same evolution chain.

dual-mem requires a JEV key, an LLM API key and an embedding API key. Missing credentials
raise on construction. Tests and custom backends inject ``jev=`` / ``llm=`` / ``embed=``.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import uuid
from typing import Any

from dual_mem.config import Settings
from dual_mem.agent.review import ReviewWorker
from dual_mem.isolation import build_filter
from dual_mem.locks import LockRegistry
from dual_mem.registry import _UNSET, ComponentFactory
from dual_mem.retrieval.reader import Reader
from dual_mem.sdk_models import (
    MemoryDecision,
    ChatMessage,
    DeleteBulkResult,
    DeleteResult,
    DigestResult,
    MemoryItem,
    ScopeSummary,
    SearchResult,
    UpdateResult,
    WriteResult,
    write_status,
)
from dual_mem.system2.system2_writer import System2Writer
from dual_mem.types import MemoryStatus
from dual_mem.writer.memory_writer import MemoryWriter

logger = logging.getLogger("dual_mem.client")

_INJECTION_BLOCK_RE = re.compile(
    r"<(relevant-memories|user-profile|memory-tools-guide|topic-catalog)"
    r"\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)


def strip_memory_injection(text: str) -> str:
    """Remove host-injected memory blocks so they are not written back as L1."""
    if not text:
        return text
    cleaned = _INJECTION_BLOCK_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


class MissingCredentialsError(RuntimeError):
    """Raised when MemoryClient is constructed without the required Jev / LLM / embedding providers."""


class MemoryClient:
    """High-level async entry point for dual-mem.

    Picks ``system1`` (MemoryWriter) or ``dual`` (System2Writer) from ``Settings.mode``.

    Constructor kwargs (see also ``Settings`` / ``~/.dual_mem/config.yaml``):

    - ``mode``: ``"system1"`` (default, L0/L1/L2 versions) or ``"dual"`` (+ explicit digest L3).
    - ``storage_dir``: on-disk root for Chroma / Kuzu / SQLite; default ``./.dual_mem_v4``.
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
        jev=None,
    ) -> None:
        overrides = {}
        if storage_dir is not None:
            overrides["storage_dir"] = storage_dir
        if mode is not None:
            overrides["mode"] = mode
        if settings is None:
            settings = Settings(**overrides)
        elif overrides:
            settings = Settings.model_validate({**settings.model_dump(), **overrides})
        self.settings = settings
        self.mode = mode or settings.mode

        self.factory = ComponentFactory(
            settings=settings,
            embed=embed,
            llm=llm if llm is not None else _UNSET,
            jev=jev,
        )

        self._validate_credentials(
            injected_embed=embed is not None, injected_llm=llm is not None
        )
        if jev is None and not settings.jev_api_key:
            raise MissingCredentialsError(
                "Jev requires DUAL_MEM_JEV_API_KEY or jev= provider"
            )
        self._build_writer()
        self.reader = Reader(factory=self.factory)
        self._write_locks = self.factory.commit_locks
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
        jev=None,
    ) -> MemoryClient:
        """Create a client from a canonical flat config dict (see ``Settings.from_dict``).

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
        return cls(settings=settings, embed=embed, llm=llm, jev=jev)

    @classmethod
    async def acreate(
        cls,
        *,
        settings: Settings | None = None,
        storage_dir: str | None = None,
        mode: str | None = None,
        embed=None,
        llm=None,
        jev=None,
    ) -> "MemoryClient":
        """Async factory mirror of ``MemoryClient(...)``; kept for API symmetry with prior versions."""
        return cls(
            settings=settings,
            storage_dir=storage_dir,
            mode=mode,
            embed=embed,
            llm=llm,
            jev=jev,
        )

    def _validate_credentials(
        self, *, injected_embed: bool, injected_llm: bool
    ) -> None:
        """Fail fast when LLM or embedding credentials are missing.

        Injected ``embed`` / ``llm`` clients (tests, custom backends) bypass the api_key check
        — the caller is responsible for those. Raises ``MissingCredentialsError`` otherwise.
        """
        missing: list[str] = []
        if not injected_llm and not self.settings.llm_api_key:
            missing.append(
                "llm_api_key (set DUAL_MEM_LLM_API_KEY or pass settings.llm_api_key)"
            )
        if not injected_embed and not self.settings.embed_api_key:
            missing.append(
                "embed_api_key (set DUAL_MEM_EMBED_API_KEY or pass settings.embed_api_key)"
            )
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
        )
        self._coding_writer = CodingWriter(
            store=store,
            extractor=extractor,
            reconciler=CodingMemoryReconciler(jev=self.factory.jev),
            llm=self.factory.llm,
            jev=self.factory.jev,
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
        request_id: str | None = None,
    ) -> WriteResult:
        """Write one memory and run the cognition pipeline (Extract -> commit -> ...).

        Pass either ``content`` (single blob) or ``messages`` (multi-turn chat) — same API.

        ``app_id`` defaults to ``settings.default_app_id`` when omitted.
        ``user_id`` is required. With ``messages``, L0_RAW and the extractor see the shaped
        full dialogue after host-injected system turns are removed.

        Cost per add: one candidate generation, then Jev target selection and (when a
        target exists) one action round; MULTIPLE targets, MERGE verification and a
        coverage regeneration each add calls. ``WriteResult.status`` reports
        committed / known / raw_only / pending_review; pending items are resolved by
        ``digest()``. Batch ``messages`` at session end to reduce cost.
        """
        # Only a caller-supplied id can be replayed; the coding store records nothing
        # for generated ids.
        idempotency_key = request_id
        request_id = request_id or str(uuid.uuid4())
        start = time.perf_counter()
        resolved_app_id = self._resolve_app_id(app_id)

        user_turn_count = 0
        if messages:
            # Coding path: check for tool messages before chat normalization
            if self._coding_writer:
                raw_dicts = [
                    (
                        dict(m)
                        if isinstance(m, dict)
                        else {"role": m.role, "content": m.content}
                    )
                    for m in messages
                    if (m.get("role") if isinstance(m, dict) else m.role) != "system"
                ]
                for raw in raw_dicts:
                    raw["content"] = strip_memory_injection(
                        str(raw.get("content") or "")
                    )
                from dual_mem.coding.preproc import (
                    has_any_tool_message,
                    strip_tool_messages,
                )

                if has_any_tool_message(raw_dicts):
                    coding_result = await self._coding_writer.write(
                        raw_dicts,
                        user_id=user_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        app_id=resolved_app_id,
                        request_id=idempotency_key,
                    )
                    if coding_result is not None:
                        mids = coding_result.get("memory_ids") or []
                        decisions = [
                            MemoryDecision(
                                action=d["action"],
                                update_type=d.get("update_type") or None,
                                targets=list(d.get("targets") or []),
                                proposal_ids=list(d.get("proposal_ids") or []),
                                source_ids=list(d.get("source_ids") or []),
                                reason=d.get("reason") or "",
                            )
                            for d in coding_result.get("decisions", [])
                        ]
                        return WriteResult(
                            success=True,
                            memory_id=coding_result["raw_id"],
                            extra_node_ids=mids,
                            extracted_count=len(mids),
                            commit_passed=bool(mids),
                            is_ephemeral=bool(decisions)
                            and all(d.action == "RAW_ONLY" for d in decisions),
                            decisions=decisions,
                            status=write_status(decisions),
                            request_id=request_id,
                            processing_time_ms=round(
                                (time.perf_counter() - start) * 1000, 2
                            ),
                        )
                    # Not coding → strip tool messages so chat path sees clean dialogue
                    messages = strip_tool_messages(raw_dicts)
            normalized = _normalize_messages(messages)
            # Host-injected role=system turns (e.g. "You are a helpful assistant") are not
            # user memory — drop before L1/extract. Extract LLM's own EXTRACT_* template is
            # separate instruction, not part of this dialogue.
            normalized = [m for m in normalized if m.role != "system"]
            normalized = [
                ChatMessage(role=m.role, content=strip_memory_injection(m.content))
                for m in normalized
            ]
            normalized = [m for m in normalized if m.content.strip()]
            cpt = self.settings.chars_per_token
            threshold_chars = int(
                self.settings.llm_context_window
                * self.settings.extract_dialogue_context_ratio
                * cpt
            )
            normalized = _shape_history(
                normalized,
                threshold_chars=threshold_chars,
                assistant_max_chars=int(
                    self.settings.extract_assistant_max_tokens * cpt
                ),
            )
            user_turn_count = sum(
                1 for m in normalized if m.role == "user" and m.content.strip()
            )
            content = _format_dialogue(normalized)
        else:
            content = strip_memory_injection(content)

        if not content.strip():
            return WriteResult(
                success=True,
                memory_id="",
                request_id=request_id,
                processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
                commit_passed=False,
                extracted_count=0,
                is_ephemeral=True,
                status="raw_only",
            )

        logger.info(
            "add app=%s user=%s mode=%s len=%d turns=%d",
            resolved_app_id,
            user_id,
            self.mode,
            len(content),
            user_turn_count,
        )

        write_kwargs = {
            "content": content,
            "app_id": resolved_app_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "request_id": request_id,
            "memory_at": memory_at,
        }
        write_kwargs["messages"] = normalized if messages is not None else None
        async with self._user_write_lock_ctx(resolved_app_id, user_id):
            result = await self.writer.write(**write_kwargs)
        return WriteResult(
            success=True,
            memory_id=result.memory_id,
            request_id=request_id,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            commit_passed=result.commit_passed,
            extracted_count=len(result.extra_node_ids),
            extra_node_ids=list(result.extra_node_ids),
            is_ephemeral=result.is_ephemeral,
            decisions=result.decisions,
            status=result.status,
            needs_review=list(result.needs_review),
        )

    async def add_raw(
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
        """Write L0_RAW only (embed + store). Does not run extract."""
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        resolved_app_id = self._resolve_app_id(app_id)
        text, _normalized, _turns = _prepare_write_text(
            content,
            messages,
            settings=self.settings,
        )
        if not text.strip():
            return WriteResult(
                success=True,
                memory_id="",
                request_id=request_id,
                processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
                commit_passed=False,
                extracted_count=0,
                is_ephemeral=True,
                status="raw_only",
            )
        async with self._user_write_lock_ctx(resolved_app_id, user_id):
            result = await self.writer.write_raw(
                content=text,
                app_id=resolved_app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_at=memory_at,
            )
        return WriteResult(
            success=True,
            memory_id=result.memory_id,
            request_id=request_id,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            commit_passed=False,
            extracted_count=0,
            is_ephemeral=True,
            status="raw_only",
        )

    async def distill(
        self,
        *,
        content: str = "",
        messages: list[dict] | list[ChatMessage] | None = None,
        user_id: str,
        source_node_ids: list[str],
        app_id: str | None = None,
        agent_id: str = "",
        session_id: str = "",
        memory_at: int | None = None,
    ) -> WriteResult:
        """Extract from existing L1 nodes without writing another L1."""
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        resolved_app_id = self._resolve_app_id(app_id)
        text, normalized, _turns = _prepare_write_text(
            content,
            messages,
            settings=self.settings,
        )
        if not text.strip() or not source_node_ids:
            return WriteResult(
                success=True,
                memory_id=source_node_ids[0] if source_node_ids else "",
                request_id=request_id,
                processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
                commit_passed=False,
                extracted_count=0,
                is_ephemeral=True,
                status="raw_only",
            )
        async with self._user_write_lock_ctx(resolved_app_id, user_id):
            result = await self.writer.distill(
                content=text,
                app_id=resolved_app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                request_id=request_id,
                source_node_ids=list(source_node_ids),
                memory_at=memory_at,
                messages=normalized,
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
            decisions=result.decisions,
            status=result.status,
            needs_review=list(result.needs_review),
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
        profile_limit: int | None = None,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        include_derived: bool | None = None,
        created_after: int | None = None,
        request_id: str | None = None,
        debug: bool = False,
        include_derived_fusion: bool = True,
        as_of: int | None = None,
        raw_limit: int = 0,
        include_relations: bool = True,
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
            limit: normal 路（L1 facts）返回上限 /
                cap on the normal route (L1 facts memories).
            min_score: normal 路最低融合分，低于此分丢弃 / min fused score for the normal route.
            profile_limit: profile 路（L2/L3 画像类）上限；省略时 System1
                关闭、dual 不限，显式 ``-1`` = 不限 / cap on the profile route;
                defaults to disabled for System1 and unlimited for dual, explicit ``-1`` = unlimited.
            profile_min_score: profile 路最低分 / min score for the profile route.
            intention_limit: proactive 路（L1/L3 意图）返回条数，``0`` = 关闭。意图是“用户接下来
                想做什么”的主动提醒，仅在需要主动推荐时才开；事实问答场景保持 0，以免意图噪声
                挤占事实召回。 / proactive route (L1/L3 intentions) size, ``0`` = OFF. Intentions
                are forward-looking "what the user plans to do" nudges — only enable for
                proactive recommendation; keep 0 for factual QA so intention noise does not
                crowd out facts.
            include_derived: 是否返回派生画像/evolution chain；省略时 System1
                为 ``False``、dual 为 ``True``。可显式覆盖。
            created_after: 只返回该 Unix 秒之后创建的记忆（时间窗过滤）/
                only memories created after this Unix-second cutoff.
            request_id: 日志/链路追踪 id，省略自动生成 / trace id for logs; auto-generated.
            debug: ``True`` 时填充 ``SearchResult.read_result`` 的逐阶段 trace /
                fill ``SearchResult.read_result`` with per-stage trace metadata.
            as_of: 有效期判定时刻（Unix 秒）。显式 ``valid_from``/``valid_until`` 不含该
                时刻的记忆（如已过期的 TEMPORAL 例外）被过滤；``None`` 不过滤。
                benchmark 传问题时间，在线场景传当前时间 / drop memories whose
                explicit validity window excludes this time.
            raw_limit: L0 原文兜底路条数，``0`` = 关闭。原文按语义 + BM25 RRF 排序，
                分数压在 0.5 以下，放在 ``SearchMemories.raw`` / fallback L0_RAW
                evidence for details extraction did not keep.
            include_relations: 为命中填充一跳 ``previous``（被直接取代的旧值）与仍有效的
                ``conflicts`` / attach the one-hop previous value and live conflicts.
        """
        request_id = request_id or str(uuid.uuid4())
        start = time.perf_counter()
        resolved_app_ids = (
            app_ids if app_ids is not None else [self.settings.default_app_id]
        )
        logger.info(
            "search app=%s user=%s query=%r limit=%d debug=%s",
            (resolved_app_ids[0] if resolved_app_ids else ""),
            user_id,
            query,
            limit,
            debug,
        )
        resolved_profile_limit = (
            (-1 if self.settings.enable_graph else 0)
            if profile_limit is None
            else profile_limit
        )
        resolved_include_derived = (
            self.settings.enable_graph if include_derived is None else include_derived
        )
        memories, trace = await self.reader.search(
            query=query,
            app_ids=resolved_app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            limit=limit,
            min_score=min_score,
            profile_limit=resolved_profile_limit,
            profile_min_score=profile_min_score,
            intention_limit=intention_limit,
            include_derived=resolved_include_derived,
            include_derived_fusion=include_derived_fusion,
            created_after=created_after,
            request_id=request_id,
            collect_trace=debug,
            as_of=as_of,
            raw_limit=raw_limit,
            include_relations=include_relations,
        )

        return SearchResult(
            success=True,
            request_id=request_id,
            memories=memories,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            read_result=trace,
        )

    async def search_conversation(
        self,
        *,
        query: str,
        app_ids: list[str] | None = None,
        user_id: str,
        agent_ids: list[str] | None = None,
        session_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        created_after: int | None = None,
        request_id: str | None = None,
    ) -> SearchResult:
        """Search L0_RAW originals (including retained history). Not used by default QA."""
        request_id = request_id or str(uuid.uuid4())
        start = time.perf_counter()
        resolved_app_ids = (
            app_ids if app_ids is not None else [self.settings.default_app_id]
        )
        memories = await self.reader.search_conversation(
            query=query,
            app_ids=resolved_app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            limit=limit,
            min_score=min_score,
            created_after=created_after,
        )
        return SearchResult(
            success=True,
            request_id=request_id,
            memories=memories,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    async def get(self, memory_id: str) -> MemoryItem | None:
        """Fetch a single memory by id, or None if it does not exist."""
        node = self.factory.vector.get(memory_id)
        if node is None:
            return None
        return Reader.memory_node_to_item(node)

    async def list(
        self,
        *,
        app_id: str | None = None,
        user_id: str,
        agent_id: str = "",
        limit: int = 100,
        include_history: bool = False,
    ) -> list[MemoryItem]:
        """List current memories under the given app/user (optionally agent) scope."""
        resolved_app_id = self._resolve_app_id(app_id)
        where = build_filter(
            app_ids=[resolved_app_id],
            user_id=user_id,
            agent_ids=[agent_id],
            statuses=[MemoryStatus.ACTIVE],
            latest_only=not include_history,
        )
        nodes = self.factory.vector.get_many(where, limit=limit)
        return [Reader.memory_node_to_item(node) for node in nodes]

    async def update(self, memory_id: str, content: str) -> UpdateResult:
        """Append an explicit user correction and return its new version ID."""
        from dataclasses import replace

        node = self.factory.vector.get(memory_id)
        if node is None:
            return UpdateResult(success=False, error_code=404)
        if not content.strip() or not node.is_latest:
            return UpdateResult(success=False, error_code=409)
        scope = (node.app_id, node.user_id, node.agent_id)
        replacement = replace(
            node,
            node_id=str(uuid.uuid4()),
            content=content,
            embedding=await self.factory.embed.embed(content),
            supersedes=[node.node_id],
            superseded_by=[],
            revision=0,
            gmt_created=int(time.time()),
            gmt_modified=None,
            s2_evidence_count=0,
            custom={
                **(node.custom or {}),
                "update_type": "OVERRIDE",
                "explicit_update": True,
            },
        )
        if not self.factory.vector.apply_version_plan(
            nodes=[replacement],
            expected={node.node_id: node.revision},
            archives=[],
            scope=scope,
        ):
            return UpdateResult(success=False, error_code=409)
        if self.mode == "dual":
            self.factory.cache.enqueue_s2_task(
                user_id=node.user_id, app_id=node.app_id, agent_id=node.agent_id
            )
        return UpdateResult(success=True, memory_id=replacement.node_id)

    async def delete(self, memory_id: str) -> DeleteResult:
        """Archive the current representation while retaining its evidence/history."""
        node = self.factory.vector.get(memory_id)
        if node is None:
            return DeleteResult(success=False, error_code=404)
        if not node.is_latest:
            return DeleteResult(success=False, error_code=409)
        if not self.factory.vector.apply_version_plan(
            nodes=[],
            expected={memory_id: node.revision},
            archives=[memory_id],
            scope=(node.app_id, node.user_id, node.agent_id),
        ):
            return DeleteResult(success=False, error_code=409)
        if self.mode == "dual":
            self.factory.cache.enqueue_s2_task(
                user_id=node.user_id, app_id=node.app_id, agent_id=node.agent_id
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
        """Review deferred write decisions, then (dual) run System2 cognition.

        Both modes re-decide memories written with an undecided relation and re-extract
        sources whose extraction was judged incomplete (``WriteResult.needs_review``).
        Failing review items stay queued and are quarantined after
        ``review_max_attempts``; they never fail the digest.
        """
        started = time.perf_counter()
        review = await ReviewWorker(factory=self.factory).run()
        timing: dict = {
            "review_sec": time.perf_counter() - started,
            "review_failed": float(review["failed"]),
        }
        if self.settings.request_outcome_ttl_seconds:
            cutoff = time.time() - self.settings.request_outcome_ttl_seconds
            self.factory.cache.purge_request_outcomes(older_than=cutoff)
            if self._coding_writer:
                self._coding_writer.store.purge_requests(older_than=cutoff)
        if not isinstance(self.writer, System2Writer):
            return DigestResult(
                success=True, processed=0, reviewed=review["reviewed"], timing=timing
            )
        processed = await self.writer.digest_pending()
        timing.update(self.writer.last_digest_stats)
        if self.settings.purge_done_queues:
            self.factory.cache.purge_done_queues()
        return DigestResult(
            success=True, processed=processed, reviewed=review["reviewed"], timing=timing
        )

    def pending_reviews(self, *, status: str = "pending") -> list[dict]:
        """Deferred decisions awaiting ``digest()`` (``status="quarantined"`` for failures)."""
        return self.factory.cache.list_reviews(status=status)

    async def search_coding(
        self,
        *,
        query: str,
        user_id: str,
        agent_id: str = "default_agent",
        app_id: str | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Search coding memories (tool-use / engineering memories)."""
        if not self._coding_writer:
            return []
        resolved = self._resolve_app_id(app_id)
        return await self._coding_writer.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            app_id=resolved,
            top_k=top_k,
        )

    async def aclose(self) -> None:
        """Release storage resources. Idempotent; call once at application shutdown."""
        if self._coding_writer is not None:
            self._coding_writer.store.close()
        provider = self.factory._jev
        if provider is not None and hasattr(provider, "aclose"):
            await provider.aclose()
        for provider in (self.factory._embed, self.factory._llm):
            if (
                provider is not _UNSET
                and provider is not None
                and hasattr(provider, "aclose")
            ):
                await provider.aclose()
        self.factory.close()


def _prepare_write_text(
    content: str,
    messages: list | None,
    *,
    settings: Settings,
) -> tuple[str, list[ChatMessage] | None, int]:
    """Strip injection, drop system turns, shape history; return (text, msgs, user turns)."""
    if not messages:
        return strip_memory_injection(content), None, 0
    normalized = _normalize_messages(messages)
    normalized = [m for m in normalized if m.role != "system"]
    normalized = [
        ChatMessage(role=m.role, content=strip_memory_injection(m.content))
        for m in normalized
    ]
    normalized = [m for m in normalized if m.content.strip()]
    cpt = settings.chars_per_token
    threshold_chars = int(
        settings.llm_context_window * settings.extract_dialogue_context_ratio * cpt
    )
    normalized = _shape_history(
        normalized,
        threshold_chars=threshold_chars,
        assistant_max_chars=int(settings.extract_assistant_max_tokens * cpt),
    )
    user_turn_count = sum(
        1 for m in normalized if m.role == "user" and m.content.strip()
    )
    return _format_dialogue(normalized), normalized, user_turn_count


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
                ChatMessage(
                    role=msg.role, content=msg.content[:assistant_max_chars] + "…"
                )
            )
    return shaped


def _format_dialogue(messages: list[ChatMessage]) -> str:
    """Render normalized messages into a natural-language dialogue chunk for L0_RAW + extract."""
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
