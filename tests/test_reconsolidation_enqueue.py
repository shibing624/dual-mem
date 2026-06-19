# -*- coding: utf-8 -*-
"""Tests for ReconsolidationHook real-enqueue path + s2_queue task_type/payload migration."""
from conftest import FakeLLMClient

from dual_mem import MemoryClient
from dual_mem.storage.cache_store import CacheStore


def _pending_rows(cache: CacheStore) -> list[dict]:
    rows = cache.conn.execute(
        "SELECT user_id, app_id, agent_id, task_type, payload "
        "FROM s2_queue WHERE status = 'pending' ORDER BY id ASC"
    ).fetchall()
    return [dict(r) for r in rows]


def test_enqueue_reconsolidation_writes_task_type_payload(tmp_storage):
    cache = CacheStore(tmp_storage)
    cache.enqueue_s2_task(
        user_id="u",
        app_id="app",
        agent_id="",
        task_type="reconsolidation",
        payload={"query": "hello", "node_ids": ["n1", "n2"]},
    )
    task = cache.dequeue_s2_task()
    assert task is not None
    assert task["task_type"] == "reconsolidation"
    assert task["payload"] == {"query": "hello", "node_ids": ["n1", "n2"]}
    assert cache.dequeue_s2_task() is None


def test_cognition_and_reconsolidation_can_coexist(tmp_storage):
    cache = CacheStore(tmp_storage)
    cache.enqueue_s2_task("u", "app")  # cognition default
    cache.enqueue_s2_task(
        "u", "app", agent_id="",
        task_type="reconsolidation",
        payload={"query": "q", "node_ids": ["n1"]},
    )
    # Same (app, user) but different task_type must NOT dedupe.
    pending = _pending_rows(cache)
    assert len(pending) == 2
    types = {row["task_type"] for row in pending}
    assert types == {"cognition", "reconsolidation"}

    # Re-enqueueing the same task_type should still dedupe.
    cache.enqueue_s2_task("u", "app")
    cache.enqueue_s2_task(
        "u", "app", agent_id="",
        task_type="reconsolidation",
        payload={"query": "q", "node_ids": ["n1"]},
    )
    assert len(_pending_rows(cache)) == 2

    # list_pending_s2_users surfaces both tuples.
    users = cache.list_pending_s2_users()
    assert {row["task_type"] for row in users} == {"cognition", "reconsolidation"}


async def test_reader_search_enqueues_reconsolidation_in_ultra(tmp_storage, fake_embed):
    """Ultra-mode search must enqueue a task_type=reconsolidation task with query+node_ids payload."""
    # Use manual trigger so the search-side fire-and-forget drain does not race with our
    # assertion (we want to observe the task while it is still pending).
    from dual_mem.config import Settings

    client = MemoryClient(
        settings=Settings(mode="dual", storage_dir=tmp_storage, system2_trigger_mode="manual"),
        embed=fake_embed,
        llm=FakeLLMClient(responses={"json": []}),
    )
    # Seed one fact directly so search has a hit (node_ids non-empty).
    from dual_mem.types import Layer, MemoryNode

    node = MemoryNode(
        content="用户喜欢喝咖啡", layer=Layer.L2_FACT, app_id="app", user_id="u", node_id="n1"
    )
    node.embedding = await fake_embed.embed("用户喜欢喝咖啡")
    client.factory.vector.upsert([node])

    # Drain anything that was enqueued during seed (none expected) before search.
    while client.factory.cache.dequeue_s2_task() is not None:
        pass

    res = await client.search(query="用户喜欢喝咖啡", app_ids=["app"], user_id="u", min_score=0.0)
    assert any(m.memory_id == "n1" for m in res.memories.normal)

    # Wait for the fire-and-forget reconsolidation hook task to land in the queue.
    import asyncio
    for _ in range(20):
        rows = _pending_rows(client.factory.cache)
        if any(r["task_type"] == "reconsolidation" for r in rows):
            break
        await asyncio.sleep(0.01)

    rows = _pending_rows(client.factory.cache)
    recon = [r for r in rows if r["task_type"] == "reconsolidation"]
    assert len(recon) >= 1
    import json
    payload = json.loads(recon[0]["payload"])
    assert payload["query"] == "用户喜欢喝咖啡"
    assert "n1" in payload["node_ids"]

    await client.aclose()


async def test_reader_search_skips_reconsolidation_in_system1(tmp_storage, fake_embed):
    """Non-dual modes must NOT enqueue a reconsolidation task (enable_graph is False)."""
    client = MemoryClient(
        storage_dir=tmp_storage,
        mode="system1",
        embed=fake_embed,
        llm=FakeLLMClient(),
    )
    from dual_mem.types import Layer, MemoryNode

    node = MemoryNode(
        content="x", layer=Layer.L2_FACT, app_id="app", user_id="u", node_id="n1"
    )
    node.embedding = await fake_embed.embed("x")
    client.factory.vector.upsert([node])

    res = await client.search(query="x", app_ids=["app"], user_id="u", min_score=0.0)
    assert any(m.memory_id == "n1" for m in res.memories.normal)

    import asyncio
    await asyncio.sleep(0.05)
    rows = _pending_rows(client.factory.cache)
    assert all(r["task_type"] != "reconsolidation" for r in rows)

    await client.aclose()
