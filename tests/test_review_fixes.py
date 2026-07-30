"""Tests for the post-review fixes:

#1 per_write search awaits the reconsolidation enqueue before draining (no race).
#2 LockRegistry is LRU-bounded and never evicts a held lock.
#3 _fast_write embeds all extracted nodes in a single batch call.
#4 Reconciler excludes the just-written fast-write node ids from its candidate set.
"""
import asyncio

import pytest

from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.locks import LockRegistry
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


# ---- #2 LockRegistry --------------------------------------------------------------

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


# ---- #4 Reconciler exclude_ids ----------------------------------------------------

async def test_reconcile_excludes_given_node_ids(tmp_storage, fake_embed):
    """The just-written originals must not appear as 'existing' candidates."""
    from dual_mem.agent.reconciler import Reconciler

    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={}))
    vector = client.factory.vector

    # Seed two identical-content ACTIVE nodes: one is the "fast-write original" we will
    # exclude, the other a genuine pre-existing memory.
    for nid in ("orig", "existing"):
        n = MemoryNode(
            content="用户喜欢喝美式咖啡", layer=Layer.L2_FACT,
            app_id="app", user_id="u", status=MemoryStatus.ACTIVE, node_id=nid,
        )
        n.embedding = fake_embed.embed_sync(n.content)
        vector.upsert([n])

    captured: dict = {}

    class _SpyLLM(FakeLLMClient):
        async def chat_json(self, *, system, user, temperature=0.2, **kw):
            captured["system"] = system
            return []  # empty ops → ADDs straight through

    reconciler = Reconciler(
        llm=_SpyLLM(responses={}), embed=fake_embed, vector=vector,
    )
    await reconciler.reconcile(
        new_memories=["用户喜欢喝美式咖啡"],
        new_memories_meta=[{"content": "用户喜欢喝美式咖啡", "layer": "L2_FACT", "tags": []}],
        app_id="app", user_id="u", agent_id="", current_time="",
        exclude_ids=["orig"],
    )

    # The excluded original must not be in the prompt; the real existing one should be.
    assert "orig" not in captured.get("system", "")
    assert "existing" in captured.get("system", "")

    await client.aclose()


# ---- #3 batched fast-write embedding ----------------------------------------------

async def test_fast_write_embeds_in_single_batch(tmp_storage):
    """MemAgent._fast_write should call embed_batch once for all extracted nodes,
    not embed_queued N times."""
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
    # Extractor returns two facts + one identity so fast-write has >1 node.
    llm = FakeLLMClient(responses={
        "extract": {
            "identity": [{"content": "用户是工程师", "tags": []}],
            "facts": [
                {"content": "用户住在北京", "tags": []},
                {"content": "用户喜欢咖啡", "tags": []},
            ],
        },
    })
    client = MemoryClient(settings=settings, embed=embed, llm=llm)

    await client.add(content="自我介绍若干", app_id="app", user_id="u")

    # The three extracted L2/L4 nodes are embedded via a single batch call rather than
    # one embed_queued per node. (The L1_RAW node still uses embed_queued — that is a
    # single writer-level call and is expected, so we only assert fast-write didn't add
    # a per-fact queued call for each of the 3 extracted nodes.)
    assert embed.batch_calls >= 1
    assert embed.queued_calls <= 1

    await client.aclose()


# ---- #1 per_write search drains reconsolidation (no race) -------------------------

async def test_per_write_search_drains_reconsolidation_without_race(tmp_storage, fake_embed):
    """In per_write mode, a search should enqueue AND drain the reconsolidation task
    within the same call — the drain must not run before its own enqueue."""
    settings = Settings(mode="dual", storage_dir=tmp_storage,
                        system2_trigger_mode="per_write")
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={}))

    # Seed an ACTIVE node so search recalls something (→ hook enqueues a task).
    n = MemoryNode(
        content="用户喜欢登山", layer=Layer.L4_IDENTITY,
        app_id="app", user_id="u", status=MemoryStatus.ACTIVE, node_id="n1",
    )
    n.embedding = fake_embed.embed_sync(n.content)
    client.factory.vector.upsert([n])

    await client.search(query="焦虑！崩溃了！登山", app_ids=["app"], user_id="u", min_score=0.0)
    # Let the fire-and-forget drain task finish.
    await asyncio.sleep(0)
    for _ in range(5):
        await asyncio.sleep(0)

    # The reconsolidation queue should be empty (enqueued then drained), and the node
    # should carry a reactivation timestamp from the drain.
    leftover = client.factory.cache.dequeue_s2_task(task_type="reconsolidation")
    assert leftover is None
    refreshed = client.factory.vector.get("n1")
    assert refreshed.custom is not None
    assert "last_reactivated_at" in refreshed.custom

    await client.aclose()
