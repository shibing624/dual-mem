from dual_mem.config import Settings
from dual_mem.providers.embedding import EmbedService
from dual_mem.providers.llm import LLMClient
from dual_mem.storage.cache_store import CacheStore
from dual_mem.storage.graph_store import KuzuGraphStore
from dual_mem.storage.history_store import HistoryStore
from dual_mem.storage.vector_store import ChromaVectorStore

_UNSET = object()


class ComponentFactory:
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
        if self._vector is None:
            self._vector = ChromaVectorStore(self.settings.storage_dir)
        return self._vector

    @property
    def cache(self) -> CacheStore:
        if self._cache is None:
            self._cache = CacheStore(self.settings.storage_dir)
        return self._cache

    @property
    def history(self) -> HistoryStore:
        if self._history is None:
            self._history = HistoryStore(self.settings.storage_dir)
        return self._history

    @property
    def graph(self) -> KuzuGraphStore | None:
        if self._graph is _UNSET:
            self._graph = (
                KuzuGraphStore(self.settings.storage_dir)
                if self.settings.enable_graph
                else None
            )
        return self._graph

    @property
    def llm(self) -> LLMClient | None:
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
