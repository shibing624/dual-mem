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
import logging
import time
import uuid

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

    - ``app_id`` (required on add): product / tenant namespace, e.g. ``"agentica"`` — not a secret.
    - ``user_id`` (required): end-user id; reads and writes must share the same pair.
    - ``app_ids`` (required on search): list, usually ``[app_id]``.
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
    ):
        if settings is None:
            overrides = {}
            if storage_dir is not None:
                overrides["storage_dir"] = storage_dir
            if mode is not None:
                overrides["mode"] = mode
            settings = Settings(**overrides)
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

    def _user_write_lock(self, app_id: str, user_id: str) -> asyncio.Lock:
        """Return the per-user write lock, creating it lazily on first access."""
        return self._write_locks.get(f"{app_id}::{user_id}")

    async def add(
        self,
        *,
        content: str = "",
        messages: list[dict] | list[ChatMessage] | None = None,
        app_id: str,
        user_id: str,
        agent_id: str = "",
        session_id: str = "",
        memory_at: int | None = None,
    ) -> WriteResult:
        """Write one memory and run the cognition pipeline (Gate → Extract → …).

        Pass either ``content`` (single blob) or ``messages`` (multi-turn chat) — same API.

        ``app_id`` and ``user_id`` are **required** isolation keys. With ``messages``,
        only ``role=='user'`` turns drive Gate vector novelty (max across turns); L1_RAW
        and the extractor still see the full dialogue; the last assistant turn feeds Gate LLM context.

        Each add costs ~2 LLM calls (Gate + Extract). For agent apps, you may batch
        ``messages`` at session end to reduce cost; ``add(content=...)`` per turn is
        also supported when low-latency persistence is required — see docs/skills tradeoff.
        """
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        user_queries: list[str] = []
        agent_context: str | None = None
        if messages:
            normalized = _normalize_messages(messages)
            user_queries = [m.content for m in normalized if m.role == "user" and m.content.strip()]
            content = _format_dialogue(normalized)
            agent_context = _last_assistant_context(normalized)

        logger.info(
            "add app=%s user=%s mode=%s len=%d turns=%d",
            app_id, user_id, self.mode, len(content), len(user_queries),
        )

        async with self._user_write_lock(app_id, user_id):
            result = await self.writer.write(
                content=content,
                app_id=app_id,
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
        app_ids: list[str],
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

        ``app_ids`` and ``user_id`` scope the query (must match values used in ``add``).
        Read path uses embedding + hybrid retrieval — no LLM. Safe to call every user turn
        in an agent loop before generation.

        ``debug=True`` fills ``SearchResult.read_result`` with per-stage trace metadata.
        """
        request_id = request_id or str(uuid.uuid4())
        start = time.perf_counter()
        logger.info(
            "search app=%s user=%s query=%r limit=%d debug=%s",
            (app_ids[0] if app_ids else ""), user_id, query, limit, debug,
        )
        if debug:
            memories, trace = await self.reader.search_with_trace(
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
            )
        else:
            memories = await self.reader.search(
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
            )
            trace = None

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
        self, *, app_id: str, user_id: str, agent_id: str = "", limit: int = 100
    ) -> list[MemoryItem]:
        """List ACTIVE memories under the given app/user (optionally agent) scope."""
        where = build_filter(
            app_ids=[app_id],
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
        app_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        confirm: bool = False,
    ) -> DeleteBulkResult:
        """Delete every memory in a scope; requires confirm=True as a safety guard."""
        if confirm is not True:
            return DeleteBulkResult(success=False, error_code=400)
        where: dict = {"app_id": {"$in": [app_id]}}
        if user_id is not None:
            where["user_id"] = user_id
        if agent_id is not None:
            where["agent_id"] = agent_id
        nodes = self.factory.vector.get_many(where)
        node_ids = [node.node_id for node in nodes]
        self.factory.vector.delete(node_ids)
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
        cores = 0
        if self.settings.cross_domain_enable:
            sweeper = CrossDomainSweeper(factory=self.factory)
            for app_id, user_id in self.writer.processed_pairs:
                result = await sweeper.run(app_id=app_id, user_id=user_id)
                if result.get("cores"):
                    cores += int(result["cores"])
        return DigestResult(success=True, processed=processed, cores_created=cores)

    async def aclose(self) -> None:
        """Release dual-mode background resources. Idempotent.

        Cancels the scheduled System2 loop when ``system2_trigger_mode="scheduled"``.
        No-op for ``system1``. Does **not** await in-flight ``per_write`` digest tasks.
        Call once at application shutdown (not after each request).
        """
        if isinstance(self.writer, System2Writer):
            await self.writer.aclose()


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
