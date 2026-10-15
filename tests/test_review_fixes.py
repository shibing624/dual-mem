"""Lock bounds, JEV candidate exclusion, and batched embedding of extracted versions."""

import pytest

from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.locks import LockRegistry
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient, FakeJev


# ---- LockRegistry -----------------------------------------------------------------


def test_lock_registry_returns_stable_lock_per_key():
    reg = LockRegistry(max_locks=8)
    a1 = reg.get("a")
    a2 = reg.get("a")
    b = reg.get("b")
    assert a1 is a2
    assert a1 is not b


def test_lock_registry_evicts_unheld_locks_over_cap():
    reg = LockRegistry(max_locks=3)
    for i in range(10):
        reg.get(f"k{i}")
    # None held → pruned back to the cap.
    assert len(reg) <= 3


async def test_lock_registry_never_evicts_a_held_lock():
    reg = LockRegistry(max_locks=2)
    held = reg.get("held")
    async with held:
        # Fill far past the cap while "held" is locked.
        for i in range(20):
            reg.get(f"other{i}")
        # The held lock must still be the same object in the registry.
        assert reg.get("held") is held
    # After release it becomes eligible for eviction again.
    for i in range(20, 40):
        reg.get(f"more{i}")
    assert len(reg) <= 2


# ---- JEV exclude_ids --------------------------------------------------------------


async def test_reconcile_excludes_given_node_ids(tmp_storage, fake_embed):
    """The just-written originals must not appear as 'existing' candidates."""
    from dual_mem.agent.reconciler import Reconciler

    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(
        settings=settings, embed=fake_embed, llm=FakeLLMClient(responses={})
    )
    vector = client.factory.vector

    # Seed two identical-content ACTIVE nodes: one is the version we will
    # exclude, the other a genuine pre-existing memory.
    for nid in ("orig", "existing"):
        n = MemoryNode(
            content="用户喜欢喝美式咖啡",
            layer=Layer.L1_FACT,
            app_id="app",
            user_id="u",
            status=MemoryStatus.ACTIVE,
            node_id=nid,
        )
        n.embedding = fake_embed.embed_sync(n.content)
        vector.upsert([n])

    jev = FakeJev()
    reconciler = Reconciler(
        jev=jev, llm=FakeLLMClient(), embed=fake_embed, vector=vector
    )
    await reconciler.reconcile(
        new_memories=["用户喜欢喝美式咖啡"],
        new_memories_meta=[{"content": "用户喜欢喝美式咖啡"}],
        app_id="app",
        user_id="u",
        agent_id="",
        current_time="",
        exclude_ids=["orig"],
    )
    assert [n["id"] for n in jev.calls[0]["state"]["current"]] == ["existing"]

    await client.aclose()


# ---- batched embedding of extracted versions --------------------------------------


async def test_extracted_versions_embed_in_single_batch(tmp_storage):
    """Extracted L1/L2 versions share one embed_batch, not one embed_queued each."""
    from conftest import FakeEmbedService

    class _CountingEmbed(FakeEmbedService):
        def __init__(self):
            super().__init__()
            self.batch_calls = 0
            self.queued_calls = 0

        async def embed_batch(self, texts):
            self.batch_calls += 1
            return await super().embed_batch(texts)

        async def embed_queued(self, text):
            self.queued_calls += 1
            return await super().embed_queued(text)

    embed = _CountingEmbed()
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    # Extractor returns two facts + one profile so the write has more than one version.
    llm = FakeLLMClient(
        responses={
            "extract": {
                "identity": [{"content": "用户是工程师", "tags": []}],
                "facts": [
                    {"content": "用户住在北京", "tags": []},
                    {"content": "用户喜欢咖啡", "tags": []},
                ],
            },
        }
    )
    client = MemoryClient(settings=settings, embed=embed, llm=llm)

    await client.add(content="自我介绍若干", app_id="app", user_id="u")

    # The extracted L1/L2 nodes share one embed_batch. L0_RAW still uses one
    # embed_queued at the writer, so queued calls stay at most one.
    assert embed.batch_calls >= 1
    assert embed.queued_calls <= 1

    await client.aclose()
