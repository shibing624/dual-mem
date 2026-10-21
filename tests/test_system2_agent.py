import json

import numpy as np
import pytest
from conftest import FakeJev, FakeLLMClient

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.system2.system2_agent import (
    CATALOG_KIND, CORE_SCHEMA_KIND, SCHEMA_KIND, System2Agent, cluster_facts, collision_groups, raw_source,
)
from dual_mem.types import Layer, MemoryNode

DIM = 64


def vec(axis: int, jitter: float = 0.0, member: int = -1) -> list[float]:
    """Unit-ish vector on ``axis``; ``member`` adds a distinct off-axis direction.

    Members of one cluster then have cosine ~0.8 (same cluster at eps 0.45, but not
    near-duplicates at 0.92).
    """
    v = [0.0] * DIM
    v[axis] = 1.0
    v[(axis + 7) % DIM] = jitter
    if member >= 0:
        v[(axis + 1 + member) % DIM] = 0.5
    return v


def create_all(user: str) -> dict:
    cluster = json.loads(user)["cluster"]
    ids = [m["id"] for m in cluster]
    anchor = cluster[0]["content"].split()[-1]
    return {"operations": [{
        "op": "create_schema",
        "content": f"When planning trips, the user books quiet stays — reflecting a need to recharge ({anchor}).",
        "evidence_ids": ids,
    }]}


def router(schema=create_all, behaviour=None, core=None, catalog=None):
    def respond(system, user):
        if "cognitive processing agent" in system:
            return schema(user)
        if "behavioral psychologist" in system:
            return behaviour(user) if behaviour else {"abstraction": "seeks calm control"}
        if "deep pattern analyst" in system:
            return core(user) if core else {"core_pattern": None}
        if "persona catalog" in system:
            return catalog(user) if catalog else {"domains": {}}
        raise AssertionError(system[:60])
    return respond


def factory(tmp_storage, embed, respond=None, **settings):
    settings.setdefault("system2_cross_domain_enabled", False)
    settings.setdefault("system2_catalog_enabled", False)
    return ComponentFactory(
        settings=Settings(mode="dual", storage_dir=tmp_storage, **settings),
        embed=embed,
        llm=FakeLLMClient({"json": respond or router()}),
        jev=FakeJev(),
    )


def fact(nid, axis, source, *, jitter=0.0, member=-1, owner="user", **kw):
    return MemoryNode(
        node_id=nid, content=f"memory {nid}", layer=Layer.L1_FACT, app_id="app", user_id="u",
        owner=owner, embedding=vec(axis, jitter, member), custom={"source_node_id": source}, **kw,
    )


def cluster_of(prefix, axis, n=4, *, sources=None, offset=0, owner="user"):
    return [fact(f"{prefix}{i}", axis, (sources or [f"{prefix}r{i}" for i in range(n)])[i],
                 member=offset + i, owner=owner) for i in range(n)]


def derived(f, kind=None):
    nodes = f.vector.get_many({"layer": Layer.L3_DERIVED.value}, limit=100)
    return [n for n in nodes if kind is None or n.kind == kind]


def test_cluster_requires_min_size_and_collapses_duplicates():
    nodes = [*cluster_of("a", 0), *cluster_of("b", 9, n=3)]
    groups = cluster_facts(nodes, distance=0.35, refine_distance=0.25, min_size=4, max_size=12, dedup_cosine=0.92)
    assert [sorted(n.node_id for n in g) for g in groups] == [["a0", "a1", "a2", "a3"]]
    dupes = [fact(f"d{i}", 3, f"r{i}", jitter=0.01 * i) for i in range(5)]
    assert cluster_facts(dupes, distance=0.35, refine_distance=0.25, min_size=4, max_size=12, dedup_cosine=0.92) == []
    assert raw_source(nodes[0]) == "ar0"


def test_oversized_cluster_is_refined_without_losing_members():
    nodes = [fact(f"x{i}", 0, f"r{i}", member=i) for i in range(6)]
    groups = cluster_facts(nodes, distance=0.35, refine_distance=0.25, min_size=2, max_size=3, dedup_cosine=0.92)
    assert groups and all(len(g) <= 3 for g in groups)


def test_collision_needs_similar_behaviour_and_different_content():
    content = np.asarray([vec(0), vec(20), vec(0, 0.1)])
    behaviour = np.asarray([vec(1), vec(1, 0.05), vec(1, 0.1)])
    # 0 and 1: same behaviour, different content -> collide. 0/2 and 1/2: content too close (0-2).
    groups = collision_groups(content, behaviour, beh_threshold=0.7, sem_threshold=0.6)
    assert sorted(map(sorted, groups)) == [[0, 1, 2]]
    assert collision_groups(content[:1], behaviour[:1], beh_threshold=0.7, sem_threshold=0.6) == []


async def test_schema_from_one_cluster_per_call_with_evidence(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed)
    f.vector.upsert([*cluster_of("a", 0), *cluster_of("b", 20), fact("c0", 40, "rc")])
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["clusters"] == 2 and stats["created_schemas"] == 2
    assert len(f.llm.calls) == 2
    for schema in derived(f, SCHEMA_KIND):
        ids = schema.custom["source_ids"]
        assert len(ids) == 4 and len(schema.custom["raw_source_ids"]) == 4
        assert f.graph.evidence_of(schema.node_id) == ids
    # Every fresh memory is reviewed once; a second digest calls no model.
    assert all(f.vector.get(n).s2_evidence_count == 1 for n in ("a0", "b3", "c0"))
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert len(f.llm.calls) == 2
    f.close()


async def test_small_or_single_source_material_calls_no_model(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed)
    f.vector.upsert(cluster_of("a", 0, n=3))
    assert (await System2Agent(factory=f).run(app_id="app", user_id="u"))["clusters"] == 0
    assert f.llm.calls == []
    f.close()


async def test_schema_needs_two_raw_sources(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed)
    f.vector.upsert(cluster_of("a", 0, sources=["same"] * 4))
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["created_schemas"] == 0 and stats["rejected_drafts"] == 1
    f.close()


async def test_assistant_memories_are_not_evidence(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed)
    f.vector.upsert(cluster_of("a", 0, owner="assistant"))
    assert (await System2Agent(factory=f).run(app_id="app", user_id="u"))["clusters"] == 0
    f.close()


@pytest.mark.parametrize("operation", [
    {"op": "create_schema", "content": "x", "evidence_ids": ["m0", "zz"]},
    {"op": "create_schema", "content": "", "evidence_ids": ["m0", "m1"]},
    {"op": "rewrite", "content": "x", "evidence_ids": ["m0", "m1"]},
])
async def test_invalid_operations_are_dropped(tmp_storage, fake_embed, operation):
    f = factory(tmp_storage, fake_embed, router(schema=lambda user: {"operations": [operation]}))
    f.vector.upsert(cluster_of("a", 0))
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["created_schemas"] == 0 and stats["rejected_drafts"] == 1
    f.close()


async def test_malformed_generation_drops_cluster_without_retry(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed, router(schema=lambda user: {"operations": "nope"}))
    f.vector.upsert(cluster_of("a", 0))
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["failed_clusters"] == 1 and len(f.llm.calls) == 1
    # The cluster was not committed, so its memories stay fresh for the next digest.
    assert f.vector.get("a0").s2_evidence_count == 0
    f.close()


async def test_new_support_is_added_as_evidence_not_a_rewrite(tmp_storage, fake_embed):
    seen = []

    def schema(user):
        payload = json.loads(user)
        seen.append(payload["existing_schemas"])
        ids = [m["id"] for m in payload["cluster"]]
        if payload["existing_schemas"]:
            return {"operations": [{"op": "add_evidence", "schema_id": "s0", "evidence_ids": ids}]}
        return create_all(user)

    f = factory(tmp_storage, fake_embed, router(schema=schema))
    f.vector.upsert(cluster_of("a", 0))
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    (first,) = derived(f, SCHEMA_KIND)
    f.vector.upsert(cluster_of("n", 0, offset=4))
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["created_schemas"] == 0 and stats["evidence_added"] == 4
    (still,) = derived(f, SCHEMA_KIND)
    assert still.content == first.content
    assert seen[1] and seen[1][0]["content"] == first.content
    assert {"n0", "n3"} <= set(f.graph.evidence_of(first.node_id))
    f.close()


async def test_superseded_memories_never_feed_cognition(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed)
    f.vector.upsert([*cluster_of("a", 0, n=3), fact("old", 0, "ro", member=3, is_latest=False)])
    assert (await System2Agent(factory=f).run(app_id="app", user_id="u"))["clusters"] == 0
    f.close()


async def test_cross_domain_core_schema_from_colliding_basics(tmp_storage, fake_embed):
    clusters = {"a": 0, "b": 12, "c": 24, "d": 36, "e": 48}  # mutually orthogonal content

    def core(user):
        assert len(json.loads(user)["schemas"]) >= 2
        return {"core_pattern": "Across domains, the user invests patiently in slow growth."}

    class AxisEmbed:
        """Schema texts get their cluster's (orthogonal) axis; the behaviour text one shared axis."""

        async def embed_batch(self, texts):
            out = []
            for t in texts:
                axis = next((a for p, a in clusters.items() if f"({p}0)" in t), 60)
                out.append(vec(axis))
            return out

    f = factory(tmp_storage, AxisEmbed(), router(core=core), system2_cross_domain_enabled=True)
    for prefix, axis in clusters.items():
        f.vector.upsert(cluster_of(prefix, axis))
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["created_schemas"] == 5
    # One shared behaviour abstraction for all five (dissimilar-content) schemas collides them.
    assert stats["core_groups"] == 1 and stats["created_core"] == 1
    (core_node,) = derived(f, CORE_SCHEMA_KIND)
    assert len(core_node.custom["member_schema_ids"]) == 5
    assert len(core_node.custom["source_ids"]) == 20
    calls = len(f.llm.calls)
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert len(f.llm.calls) == calls  # abstractions reused, the group already has a core
    f.close()


async def test_persona_catalog_selects_exact_current_facts(tmp_storage, fake_embed):
    def catalog(user):
        assert user.startswith("Recorded facts:")
        return {"domains": {"habits_items": [1, 2, 99], "values_coping": [2, 3], "unknown": [4]}}

    f = factory(tmp_storage, fake_embed, router(catalog=catalog), system2_catalog_enabled=True,
                system2_catalog_min_facts=3)
    f.vector.upsert([fact("f1", 0, "r1"), fact("f2", 10, "r2"), fact("f3", 20, "r3")])
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["catalog_entries"] == 3
    (node,) = derived(f, CATALOG_KIND)
    assert node.content.splitlines() == [
        "- [habits_items] memory f1", "- [habits_items] memory f2", "- [values_coping] memory f3",
    ]
    calls = len(f.llm.calls)
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert len(f.llm.calls) == calls  # unchanged fact bank: catalog reused
    f.vector.upsert([fact("f4", 30, "r4")])
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    current = [n for n in derived(f, CATALOG_KIND) if n.is_latest]
    assert len(current) == 1 and current[0].supersedes == [node.node_id]
    f.close()
