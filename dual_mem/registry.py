# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ComponentFactory for dependency wiring; lazily constructs and caches
providers and stores (embed/vector/cache/history/graph/llm) from Settings. dual-mem requires
both LLM and embedding API keys; the factory always materializes an LLM client and never
operates in an embedding-only / no-LLM state.
"""
from dual_mem.config import Settings, ensure_storage_dir
from dual_mem.providers.embedding import EmbedService
from dual_mem.providers.llm import LLMClient
from dual_mem.storage.cache_store import CacheStore
from dual_mem.storage.graph_store import KuzuGraphStore
from dual_mem.storage.history_store import HistoryStore
from dual_mem.storage.vector_store import ChromaVectorStore

_UNSET = object()


class ComponentFactory:
    """Lazily build and cache providers/stores; allows injecting fake embed/llm in tests."""

    def __init__(
        self,
        *,
        settings: Settings,
        embed: EmbedService | None = None,
        llm=_UNSET,
    ):
        self.settings = settings
        ensure_storage_dir(settings.storage_dir)
        self._embed: EmbedService | None = embed
        self._vector: ChromaVectorStore | None = None
        self._cache: CacheStore | None = None
        self._history: HistoryStore | None = None
        self._graph = _UNSET
        self._llm = llm
        # True when the caller injected an LLM client (tests / custom backends). Such a
        # client is trusted as-is and bypasses the api_key/probe checks done by MemoryClient.
        self.has_user_llm = llm is not _UNSET and llm is not None

    @property
    def embed(self) -> EmbedService:
        """The embedding service, constructed from Settings on first access."""
        if self._embed is None:
            self._embed = EmbedService(
                base_url=self.settings.embed_base_url,
                api_key=self.settings.embed_api_key,
                model=self.settings.embed_model,
                dim=self.settings.embed_dim,
                timeout=self.settings.embed_timeout,
                queue_batch_size=self.settings.embed_queue_batch_size,
                queue_batch_window_ms=self.settings.embed_queue_window_ms,
                input_max_tokens=self.settings.embed_max_tokens,
                chars_per_token=self.settings.chars_per_token,
                retry_attempts=self.settings.embed_retry_attempts,
                retry_base_delay=self.settings.embed_retry_base_delay,
                cache_size=self.settings.embed_cache_size,
            )
        return self._embed

    @property
    def vector(self) -> ChromaVectorStore:
        """The Chroma vector store, constructed on first access."""
        if self._vector is None:
            self._vector = ChromaVectorStore(self.settings.storage_dir)
        return self._vector

    @property
    def cache(self) -> CacheStore:
        """The cache store, constructed on first access."""
        if self._cache is None:
            self._cache = CacheStore(self.settings.storage_dir)
        return self._cache

    @property
    def history(self) -> HistoryStore:
        """The history/audit store, constructed on first access."""
        if self._history is None:
            self._history = HistoryStore(
                self.settings.storage_dir,
                persist=self.settings.persist_history,
            )
        return self._history

    @property
    def graph(self) -> KuzuGraphStore | None:
        """The graph store when graph is enabled (``dual``), else None."""
        if self._graph is _UNSET:
            self._graph = (
                KuzuGraphStore(self.settings.storage_dir)
                if self.settings.enable_graph
                else None
            )
        return self._graph

    @property
    def llm(self) -> LLMClient:
        """The LLM client. Always non-None; dual-mem requires an LLM API key."""
        if self._llm is _UNSET:
            self._llm = LLMClient(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model=self.settings.llm_model,
                timeout=self.settings.llm_timeout,
                json_mode=self.settings.llm_json_mode,
                extra_body=self.settings.llm_extra_body or None,
                input_max_chars=self.settings.llm_input_max_chars,
            )
        return self._llm
