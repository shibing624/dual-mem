"""L1 queued embedding and post-extract structured-memory batching."""
from dual_mem.agent.mem_agent import MemAgent
from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryNode
from dual_mem.writer.memory_writer import MemoryWriter

from conftest import FakeEmbedService, FakeLLMClient


class CountingEmbed(FakeEmbedService):
    def __init__(self, dim: int = 64):
        super().__init__(dim=dim)
        self.batch_calls: list[list[str]] = []
        self.queued_calls: list[str] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        return await super().embed_batch(texts)

    async def embed_queued(self, text: str) -> list[float]:
        self.queued_calls.append(text)
        return await super().embed_queued(text)


def _extract_response():
    return {
        "is_ephemeral": False,
        "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
        "identity": [{"content": "用户喜欢 Python", "speculate": None, "tags": []}],
        "facts": [],
        "intentions": [],
        "basic_info": {"location": "北京"},
    }


async def test_post_extract_single_embed_batch(tmp_storage):
    embed = CountingEmbed()
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=embed,
        llm=FakeLLMClient(responses={"extract": _extract_response(), "search_query": []}),
    )
    agent = MemAgent(factory=factory)
    raw = MemoryNode(
        content="我搬到北京了，主要用 Python",
        layer=Layer.L1_RAW,
        app_id="app",
        user_id="u",
        agent_id="ag",
    )
    raw.embedding = embed.embed_sync(raw.content)

    await agent.run(
        content=raw.content,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req",
        memory_at=None,
    )

    # L0 + one L4 in one post-extract batch (no separate basic_profile embed).
    assert len(embed.batch_calls) == 1
    assert len(embed.batch_calls[0]) == 2


async def test_writer_uses_queued_embedding_for_l1(tmp_storage):
    embed = CountingEmbed()
    factory = ComponentFactory(
        settings=Settings(
            mode="system1",
            storage_dir=tmp_storage,
        ),
        embed=embed,
        llm=FakeLLMClient(responses={"extract": _extract_response(), "search_query": []}),
    )
    writer = MemoryWriter(factory=factory)
    dialogue = "[user]: 我搬到北京\n[assistant]: 好的"

    await writer.write(
        content=dialogue,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req",
    )

    assert len(embed.queued_calls) == 1
    assert embed.queued_calls[0] == dialogue


def _basic_info_only_response():
    """Extract result with ONLY basic_info — no facts/identity/intentions."""
    return {
        "is_ephemeral": False,
        "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
        "identity": [],
        "facts": [],
        "intentions": [],
        "basic_info": {"location": "北京"},
    }


async def test_reconcile_sync_basic_info_only_writes_l0_no_enqueue(tmp_storage):
    """reconcile_sync=True with ONLY basic_info: L0 is written and nothing is queued for
    reconcile (L0 evolves via its own supersede chain, never the reconcile queue)."""
    embed = CountingEmbed()
    factory = ComponentFactory(
        settings=Settings(
            mode="system1", storage_dir=tmp_storage,
            reconcile_sync=True,
        ),
        embed=embed,
        llm=FakeLLMClient(responses={"extract": _basic_info_only_response()}),
    )
    agent = MemAgent(factory=factory)
    raw = MemoryNode(
        content="我搬到北京了", layer=Layer.L1_RAW,
        app_id="app", user_id="u", agent_id="ag",
    )
    raw.embedding = embed.embed_sync(raw.content)

    stored_ids, _commit, _eph = await agent.run(
        content=raw.content,
        app_id="app", user_id="u", agent_id="ag", session_id="se",
        request_id="req", memory_at=None,
    )

    # Exactly one L0 node persisted.
    assert len(stored_ids) == 1
    l0 = factory.vector.get(stored_ids[0])
    assert l0 is not None and l0.layer is Layer.L0_BASIC_INFO
    # No reconcile task enqueued for an L0-only write.
    assert factory.cache.dequeue_reconcile_task(app_id="app", user_id="u", agent_id="ag") is None


async def test_async_path_enqueues_only_l2l4_not_l0(tmp_storage):
    """Default (async) path: the reconcile task carries L2/L4 ids only, never the L0 id."""
    embed = CountingEmbed()
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=embed,
        llm=FakeLLMClient(responses={"extract": _extract_response()}),  # L4 + basic_info
    )
    agent = MemAgent(factory=factory)
    raw = MemoryNode(
        content="我搬到北京，主要用 Python", layer=Layer.L1_RAW,
        app_id="app", user_id="u", agent_id="ag",
    )
    raw.embedding = embed.embed_sync(raw.content)

    stored_ids, _g, _e = await agent.run(
        content=raw.content,
        app_id="app", user_id="u", agent_id="ag", session_id="se",
        request_id="req", memory_at=None,
    )

    l0_ids = {nid for nid in stored_ids
              if (n := factory.vector.get(nid)) is not None and n.layer is Layer.L0_BASIC_INFO}
    task = factory.cache.dequeue_reconcile_task(app_id="app", user_id="u", agent_id="ag")
    assert task is not None
    enqueued = set(task["node_ids"])
    assert enqueued  # L4 identity was queued
    assert enqueued.isdisjoint(l0_ids)  # but the L0 id never enters the reconcile queue
