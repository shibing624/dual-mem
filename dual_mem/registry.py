# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ComponentFactory for dependency wiring; lazily constructs and caches
providers and stores (embed/vector/cache/history/graph/llm) from Settings.
"""
from dual_mem.config import Settings
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
        self._embed: EmbedService | None = embed
        self._vector: ChromaVectorStore | None = None
        self._cache: CacheStore | None = None
        self._history: HistoryStore | None = None
        self._graph = _UNSET
        self._llm = llm

    @property
    def embed(self) -> EmbedService:
        """The embedding service, constructed from Settings on first access."""
        if self._embed is None:
            self._embed = EmbedService(
                base_url=self.settings.embed_base_url,
                api_key=self.settings.embed_api_key,
                model=self.settings.embed_model,
                dim=self.settings.embed_dim,
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
            self._history = HistoryStore(self.settings.storage_dir)
        return self._history

    @property
    def graph(self) -> KuzuGraphStore | None:
        """The graph store when graph is enabled (ultra), else None."""
        if self._graph is _UNSET:
            self._graph = (
                KuzuGraphStore(self.settings.storage_dir)
                if self.settings.enable_graph
                else None
            )
        return self._graph

    @property
    def llm(self) -> LLMClient | None:
        """The LLM client for pro/ultra modes, else None (lite)."""
        if self._llm is _UNSET:
            self._llm = (
                LLMClient(
                    base_url=self.settings.llm_base_url,
                    api_key=self.settings.llm_api_key,
                    model=self.settings.llm_model,
                )
                if self.settings.mode in ("pro", "ultra")
                else None
            )
        return self._llm
