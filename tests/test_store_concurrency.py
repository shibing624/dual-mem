# -*- coding: utf-8 -*-
"""Embedded stores must tolerate asyncio.to_thread parallel access."""

from __future__ import annotations

import asyncio

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode, KuzuGraphStore
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
