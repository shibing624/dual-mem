# -*- coding: utf-8 -*-
"""Hybrid integration: calibrated lexical recall, keyword rescue and dual-mode L3."""
import math

from conftest import FakeLLMClient

from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.storage.graph_store import GraphNode
from dual_mem.retrieval.hybrid_engine import _source_excerpt
from dual_mem.types import Layer, MemoryNode, MemoryStatus


def _unit(cosine: float, dim: int = 64) -> list[float]:
    """A unit vector whose dot product with [1, 0, ...] equals ``cosine``."""
    v = [0.0] * dim
    v[0] = cosine
    v[1] = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return v


_QUERY_VEC = _unit(1.0)  # == [1, 0, 0, ...]


class _ScriptedEmbed:
    """Returns a fixed vector for the search query; hashes anything else (unused here)."""

    def __init__(self, query_text: str):
        self._query_text = query_text
        self.direct_calls = 0
        self.queued_calls = 0

    async def embed(self, text: str) -> list[float]:
        self.direct_calls += 1
        return self.embed_sync(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_sync(t) for t in texts]

    async def embed_queued(self, text: str) -> list[float]:
        self.queued_calls += 1
        return self.embed_sync(text)

    def embed_sync(self, text: str) -> list[float]:
        if text == self._query_text:
            return list(_QUERY_VEC)
        return _unit(0.0)


def _seed_fact(client, *, node_id: str, content: str, cosine: float) -> None:
    node = MemoryNode(
        content=content,
        layer=Layer.L1_FACT,
        app_id="app",
        user_id="u",
        node_id=node_id,
        status=MemoryStatus.ACTIVE,
    )
    node.embedding = _unit(cosine)
    client.factory.vector.upsert([node])


async def test_bm25_rerank_surfaces_exact_term_over_higher_semantic(
    tmp_storage, monkeypatch
):
    """A keyword-matching fact outranks a *more* semantically similar distractor.

    Pure semantic would rank the distractor (cos 0.7) above the keyword hit (cos 0.5); the
    in-pool BM25 rerank (0.5*0.6 + 1.0*0.4 = 0.7 vs 0.7*0.6 = 0.42) flips that order.
    """
    query = "vault access code 70355"
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    embed = _ScriptedEmbed(query)
    client = MemoryClient(settings=settings, embed=embed, llm=FakeLLMClient())

    _seed_fact(
        client,
        node_id="kw",
        content="the vault access code is 70355 as requested",
        cosine=0.5,
    )
    _seed_fact(
        client,
        node_id="distractor",
        content="general notes about safety and storage habits",
        cosine=0.7,
    )

    query_calls = []
    get_calls = []
    original_query = client.factory.vector.query
    original_get_by_ids = client.factory.vector.get_by_ids

    def _counted_query(*args, **kwargs):
        query_calls.append((args, kwargs))
        return original_query(*args, **kwargs)

    def _counted_get_by_ids(*args, **kwargs):
        get_calls.append((args, kwargs))
        return original_get_by_ids(*args, **kwargs)

    monkeypatch.setattr(client.factory.vector, "query", _counted_query)
    monkeypatch.setattr(client.factory.vector, "get_by_ids", _counted_get_by_ids)

    result = await client.search(
        query=query, app_ids=["app"], user_id="u", min_score=0.0
    )
    normal_ids = [m.memory_id for m in result.memories.normal]
    assert "kw" in normal_ids and "distractor" in normal_ids
    assert normal_ids.index("kw") < normal_ids.index("distractor")
    assert embed.direct_calls == 1
    assert embed.queued_calls == 0
    assert len(query_calls) == 1
    assert (
        query_calls[0][1]["top_k"] == 75
    )  # v10: vdb_sem_limit = max(ceil(10*3*2.5),60) = 75
    assert get_calls == []

    await client.aclose()


async def test_lexical_recall_outside_semantic_pool_uses_calibrated_scores(tmp_storage):
    query = "vault access code 70355"
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=_ScriptedEmbed(query),
        llm=FakeLLMClient(),
    )
    nodes = [
        MemoryNode(
            node_id=f"background-{i}",
            content=f"unrelated safety notes {i}",
            layer=Layer.L1_FACT,
            app_id="app",
            user_id="u",
            embedding=_unit(0.8),
        )
        for i in range(80)
    ]
    nodes.append(
        MemoryNode(
            node_id="exact",
            content="the vault access code is 70355",
            layer=Layer.L1_FACT,
            app_id="app",
            user_id="u",
            embedding=_unit(0.35),
        )
    )
    client.factory.vector.upsert(nodes)
    result = await client.search(query=query, app_ids=["app"], user_id="u", limit=10)
    assert result.memories.normal[0].memory_id == "exact"
    assert 0.60 < result.memories.normal[0].score < 0.62
    assert all(0 <= m.score <= 1 for m in result.memories.normal)
    await client.aclose()


async def test_lexical_ubiquitous_term_outside_pool_does_not_bypass_gate(tmp_storage):
    query = "status"
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=_ScriptedEmbed(query),
        llm=FakeLLMClient(),
    )
    client.factory.vector.upsert([
        MemoryNode(
            node_id=f"status-{i}",
            content=f"status update {i}",
            layer=Layer.L1_FACT,
            app_id="app",
            user_id="u",
            embedding=_unit(0.3 if i < 80 else 0.1),
        )
        for i in range(81)
    ])
    result = await client.search(query=query, app_ids=["app"], user_id="u", limit=10)
    assert result.memories.normal == []
    await client.aclose()


async def test_i3_keyword_hit_survives_min_score_gate(tmp_storage):
    """I3: a weak-semantic (0.3) but exact-term hit clears the default 0.4 floor via fusion.

    fused = 0.3*0.6 + 1.0*0.4 = 0.58 >= 0.4 (kept); a no-keyword node at the same 0.3
    semantic has fused 0.18 < 0.4 (dropped).
    """
    query = "membership number 88231"
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(
        settings=settings, embed=_ScriptedEmbed(query), llm=FakeLLMClient()
    )

    _seed_fact(
        client, node_id="kw", content="the membership number is 88231", cosine=0.3
    )
    _seed_fact(
        client, node_id="nokw", content="some unrelated background details", cosine=0.3
    )

    result = await client.search(
        query=query, app_ids=["app"], user_id="u"
    )  # min_score=0.4
    normal_ids = [m.memory_id for m in result.memories.normal]
    assert "kw" in normal_ids
    assert "nokw" not in normal_ids

    await client.aclose()


async def test_ubiquitous_term_does_not_force_weak_hits_through_gate(tmp_storage):
    query = "status"
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(
        settings=settings,
        embed=_ScriptedEmbed(query),
        llm=FakeLLMClient(),
    )

    _seed_fact(
        client, node_id="one", content="project status meeting notes", cosine=0.3
    )
    _seed_fact(
        client, node_id="two", content="order status remains unchanged", cosine=0.3
    )

    result = await client.search(query=query, app_ids=["app"], user_id="u")
    assert result.memories.normal == []

    await client.aclose()


async def test_dual_l3_derived_recalled_into_profile(tmp_storage, fake_embed):
    """Dual mode: an L3 derived node with evidence surfaces in the profile route."""
    settings = Settings(mode="dual", storage_dir=tmp_storage)
    client = MemoryClient(settings=settings, embed=fake_embed, llm=FakeLLMClient())

    fact = MemoryNode(
        content="用户在上海的人工智能实验室做研究",
        layer=Layer.L1_FACT,
        app_id="app",
        user_id="u",
        node_id="f1",
        status=MemoryStatus.ACTIVE,
    )
    fact.embedding = fake_embed.embed_sync(fact.content)
    client.factory.vector.upsert([fact])

    schema = GraphNode(
        node_id="s1",
        layer=Layer.L3_DERIVED.value,
        content="用户从事人工智能研究",
        app_id="app",
        user_id="u",
        embedding=fake_embed.embed_sync("用户从事人工智能研究"),
        tags=["研究"],
    )
    client.factory.vector.upsert(
        [
            MemoryNode(
                node_id=schema.node_id,
                content=schema.content,
                layer=Layer.L3_DERIVED,
                kind="schema",
                app_id=schema.app_id,
                user_id=schema.user_id,
                embedding=schema.embedding,
                custom={"source_ids": ["f1"]},
            )
        ]
    )
    client.factory.graph.add_node(schema)
    client.factory.graph.add_evidence(schema_id="s1", fact_id="f1")

    result = await client.search(
        query="人工智能研究", app_ids=["app"], user_id="u", min_score=0.0
    )
    profile_ids = [m.memory_id for m in result.memories.profile]
    normal_ids = [m.memory_id for m in result.memories.normal]
    assert "s1" in profile_ids
    assert "f1" in normal_ids

    await client.aclose()


async def test_profile_route_does_not_overwrite_hybrid_score(tmp_storage):
    query = "membership number 88231"
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=_ScriptedEmbed(query), llm=FakeLLMClient(),
    )
    _seed_fact(client, node_id="fact", content="membership number is 88231", cosine=0.5)
    client.factory.vector.upsert([
        MemoryNode(node_id="profile", content="enjoys hiking and swimming",
                   layer=Layer.L2_PROFILE, app_id="app", user_id="u", embedding=_unit(0.7))
    ])
    result = await client.search(query=query, app_ids=["app"], user_id="u",
                                 profile_limit=-1, min_score=0)
    assert result.memories.flatten()[0].memory_id == "fact"
    profile = next(m for m in result.memories.profile if m.memory_id == "profile")
    assert abs(profile.score - 0.42) < 1e-5
    await client.aclose()


async def test_derived_relevance_survives_flatten_without_a_quota(tmp_storage):
    query = "technical hobbies"
    client = MemoryClient(
        settings=Settings(mode="dual", storage_dir=tmp_storage),
        embed=_ScriptedEmbed(query), llm=FakeLLMClient(),
    )
    _seed_fact(client, node_id="evidence", content="builds robots on weekends", cosine=0.6)
    derived = MemoryNode(node_id="derived", content="prefers engineering projects",
                         layer=Layer.L3_DERIVED, kind="preference", app_id="app", user_id="u",
                         embedding=_unit(0.9), custom={"source_ids": ["evidence"]})
    client.factory.vector.upsert([derived])
    client.factory.graph.add_node(GraphNode(node_id=derived.node_id, content=derived.content,
                                           layer=derived.layer.value, app_id="app", user_id="u",
                                           embedding=derived.embedding))
    client.factory.graph.add_evidence(schema_id="derived", fact_id="evidence")
    result = await client.search(query=query, app_ids=["app"], user_id="u", min_score=0,
                                 limit=1, include_derived=True)
    assert result.memories.flatten(limit=1)[0].memory_id == "derived"
    assert result.memories.flatten(limit=1)[0].score > 0.85
    off = await client.search(query=query, app_ids=["app"], user_id="u", min_score=0,
                             limit=1, include_derived=False)
    assert "derived" not in [m.memory_id for m in off.memories.flatten()]
    await client.aclose()


def test_source_excerpt_preserves_late_answer_and_speaker_context():
    content = "[user]: General meeting notes.\n[assistant]: " + "unrelated background " * 300
    content += "\n[user]: What is the return authorization for the damaged amplifier?\n"
    content += "[assistant]: The return authorization is RMA-81273; send it within 14 days.\n"
    content += "[user]: Thank you.\n[assistant]: " + "more unrelated notes " * 200
    excerpt = _source_excerpt(content, "amplifier return authorization", [])
    assert len(excerpt) <= 3200
    assert "[user]: What is the return authorization" in excerpt
    assert "[assistant]: The return authorization is RMA-81273" in excerpt
    assert "14 days" in excerpt


async def test_raw_source_link_recovers_paraphrased_evidence_without_changing_store(tmp_storage):
    query = "favorite badge color"
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=_ScriptedEmbed(query), llm=FakeLLMClient(),
    )
    source = MemoryNode(node_id="source", content="[user]: I picked the teal one.",
                        layer=Layer.L0_RAW, app_id="app", user_id="u", embedding=_unit(0.1))
    fact = MemoryNode(node_id="fact", content="favorite badge color is teal",
                      layer=Layer.L1_FACT, app_id="app", user_id="u", embedding=_unit(0.9),
                      custom={"source_node_id": "source"})
    noise = [MemoryNode(node_id=f"raw-{i}", content=f"unrelated conversation {i}",
                        layer=Layer.L0_RAW, app_id="app", user_id="u", embedding=_unit(0.8))
             for i in range(20)]
    client.factory.vector.upsert([source, fact, *noise])
    result = await client.search(query=query, app_ids=["app"], user_id="u", min_score=0,
                                 raw_limit=2)
    assert "source" in {m.memory_id for m in result.memories.raw}
    assert client.factory.vector.get("source").content == source.content
    await client.aclose()
