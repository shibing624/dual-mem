# -*- coding: utf-8 -*-
"""Embedded stores must tolerate asyncio.to_thread parallel access."""

from __future__ import annotations

import asyncio

import pytest

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.storage.cache_store import CacheStore
from dual_mem.storage.graph_store import GraphNode, KuzuGraphStore
from dual_mem.storage.history_store import HistoryStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus


async def test_chroma_parallel_query_under_lock(tmp_storage, fake_embed):
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    node = MemoryNode(
        content="parallel chroma query",
        layer=Layer.L2_FACT,
        app_id="app",
        user_id="u",
        status=MemoryStatus.ACTIVE,
    )
    node.embedding = fake_embed.embed_sync(node.content)
    factory.vector.upsert([node])

    where = {"user_id": "u", "app_id": "app", "status": MemoryStatus.ACTIVE.value}

    async def _query() -> int:
        hits = await asyncio.to_thread(
            factory.vector.query,
            embedding=node.embedding,
            where=where,
            top_k=5,
        )
        return len(hits)

    counts = await asyncio.gather(*[_query() for _ in range(8)])
    assert all(c >= 1 for c in counts)


async def test_mark_superseded_is_atomic_and_keeps_embedding(tmp_storage, fake_embed):
    """mark_superseded flips is_latest/status, appends superseded_by, preserves the embedding."""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    old = MemoryNode(
        content="老事实", layer=Layer.L2_FACT, app_id="app", user_id="u",
        status=MemoryStatus.ACTIVE, is_latest=True,
    )
    old.embedding = fake_embed.embed_sync(old.content)
    factory.vector.upsert([old])

    ok = factory.vector.mark_superseded(old.node_id, superseded_by_id="new-1")
    assert ok is True
    # Idempotent: appending the same superseder twice does not duplicate.
    assert factory.vector.mark_superseded(old.node_id, superseded_by_id="new-1") is True

    reloaded = factory.vector.get(old.node_id)
    assert reloaded.is_latest is False
    assert reloaded.status is MemoryStatus.SUPERSEDED
    assert reloaded.superseded_by == ["new-1"]
    # embedding never rewritten (float32 round-trip, so compare approximately)
    assert reloaded.embedding == pytest.approx(old.embedding, rel=1e-4)

    assert factory.vector.mark_superseded("does-not-exist", superseded_by_id="x") is False


async def test_kuzu_parallel_query_under_lock(tmp_storage):
    graph = KuzuGraphStore(tmp_storage)
    graph.add_node(
        GraphNode(
            node_id="n1",
            layer=Layer.L6_SCHEMA.value,
            content="schema one",
            app_id="app",
            user_id="u",
            embedding=[1.0, 0.0],
            gmt_created=1,
        )
    )
    graph.add_node(
        GraphNode(
            node_id="n2",
            layer=Layer.L7_INTENTION.value,
            content="intention one",
            app_id="app",
            user_id="u",
            embedding=[0.0, 1.0],
            gmt_created=2,
        )
    )

    async def _schema() -> int:
        hits = await asyncio.to_thread(
            graph.query_by_embedding,
            layer=Layer.L6_SCHEMA.value,
            user_id="u",
            app_ids=["app"],
            embedding=[1.0, 0.0],
            top_k=5,
        )
        return len(hits)

    async def _intention() -> int:
        hits = await asyncio.to_thread(
            graph.query_by_embedding,
            layer=Layer.L7_INTENTION.value,
            user_id="u",
            app_ids=["app"],
            embedding=[0.0, 1.0],
            top_k=5,
        )
        return len(hits)

    schema_n, intention_n = await asyncio.gather(_schema(), _intention())
    assert schema_n == 1
    assert intention_n == 1


async def test_cache_parallel_writes_under_lock(tmp_storage):
    cache = CacheStore(tmp_storage)

    async def _touch(i: int) -> None:
        await asyncio.to_thread(
            cache.log_pipeline,
            request_id=f"req-{i}",
            stage="TEST",
            payload={"i": i},
        )
        await asyncio.to_thread(cache.bump_access, [f"node-{i}"])
        await asyncio.to_thread(
            cache.enqueue_reconcile_task,
            app_id="app",
            user_id="u",
            agent_id="ag",
            node_ids=[f"node-{i}"],
        )

    await asyncio.gather(*[_touch(i) for i in range(16)])
    assert cache.reconcile_queue_size() == 16


async def test_history_parallel_append_under_lock(tmp_storage):
    history = HistoryStore(tmp_storage, persist=True)

    async def _append(i: int) -> None:
        await asyncio.to_thread(
            history.append,
            event="ADD",
            node_id=f"n{i}",
            user_id="u",
            old=None,
            new={"i": i},
        )

    await asyncio.gather(*[_append(i) for i in range(16)])
    rows = history.list_for_node("n0")
    assert len(rows) == 1
    assert rows[0]["new"]["i"] == 0
