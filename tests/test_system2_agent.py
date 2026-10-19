import json
import pytest
from conftest import FakeLLMClient, FakeJev
from dual_mem.registry import ComponentFactory
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode
from dual_mem.system2.system2_agent import System2Agent


def factory(tmp_storage, embed, response=None, jev=None, cap=2):
    def default(system, user):
        facts = json.loads(user)["facts"]
        ids = [f["id"] for f in facts]
        return {
            "memories": [
                {
                    "content": "derived " + ",".join(f["content"] for f in facts),
                    "kind": "schema",
                    "evidence_ids": ids,
                }
            ],
            "processed_ids": ids,
        }

    return ComponentFactory(
        settings=Settings(
            mode="dual", storage_dir=tmp_storage, system2_max_prompt_facts=cap
        ),
        embed=embed,
        llm=FakeLLMClient({"json": response or default}),
        jev=jev or FakeJev(),
    )


def seed(f, count=5):
    nodes = [
        MemoryNode(
            node_id=str(i),
            content=f"fact {i}",
            layer=Layer.L1_FACT,
            app_id="app",
            user_id="u",
            embedding=f.embed.embed_sync(f"fact {i}"),
        )
        for i in range(count)
    ]
    f.vector.upsert(nodes)
    return nodes


async def test_all_facts_paged_with_real_evidence(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed)
    original = seed(f)
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["created_schemas"] == 3
    # One generation + one validation per batch; no relation decisions for L3 nodes.
    assert len(f.llm.calls) == 3 and len(f.jev.calls) == 3
    derived = f.vector.get_many({"layer": Layer.L3_DERIVED.value}, limit=100)
    assert {nid for n in derived for nid in n.custom["source_ids"]} == {
        n.node_id for n in original
    }
    assert all(f.vector.get(n.custom["source_node_id"]) for n in derived)
    assert all(f.vector.get(n.node_id).s2_evidence_count == 1 for n in original)
    assert all(
        f.graph.evidence_of(n.node_id) == n.custom["source_ids"] for n in derived
    )
    calls = len(f.llm.calls)
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert len(f.llm.calls) == calls
    f.close()


@pytest.mark.parametrize(
    "response",
    [
        {"memories": "invalid", "processed_ids": ["0"]},
        {
            "memories": [{"content": "invented", "evidence_ids": ["foreign"]}],
            "processed_ids": ["0"],
        },
        {"memories": [{"content": "", "evidence_ids": ["0"]}], "processed_ids": ["0"]},
    ],
)
async def test_invalid_generation_is_dropped_not_retried(
    tmp_storage, fake_embed, response
):
    """Malformed generation drops that batch once; nothing is queued or retried."""
    f = factory(tmp_storage, fake_embed, response)
    seed(f, 1)
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["failed_batches"] == 1 and stats["batches"] == 1
    assert f.vector.get("0").s2_evidence_count == 0
    assert f.vector.get_many({"layer": Layer.L3_DERIVED.value}) == []
    # Second digest retries naturally because the fact was never marked processed.
    calls = len(f.llm.calls)
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert len(f.llm.calls) == calls + 1
    f.close()


async def test_rejected_draft_is_dropped_but_batch_still_commits(
    tmp_storage, fake_embed
):
    def response(system, user):
        ids = [x["id"] for x in json.loads(user)["facts"]]
        return {
            "memories": [
                {"content": "good " + ",".join(ids), "kind": "schema", "evidence_ids": ids},
                {"content": "bad " + ",".join(ids), "kind": "schema", "evidence_ids": ids},
            ],
            "processed_ids": ids,
        }

    f = factory(tmp_storage, fake_embed, response, cap=4, jev=FakeJev({"draft:1": "INVALID"}))
    seed(f, 2)
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["failed_batches"] == 0 and stats["created_schemas"] == 1
    (kept,) = f.vector.get_many({"layer": Layer.L3_DERIVED.value})
    assert kept.content.startswith("good")
    assert all(f.vector.get(n).s2_evidence_count == 1 for n in ("0", "1"))
    f.close()


async def test_preference_inference_is_versioned_derived_memory(tmp_storage, fake_embed):
    def response(system, user):
        ids = [x["id"] for x in json.loads(user)["facts"]]
        assert "kind=preference" in system
        return {
            "memories": [
                {"content": "User seems to prefer quiet venues", "kind": "preference", "evidence_ids": ids}
            ],
            "processed_ids": ids,
        }

    f = factory(tmp_storage, fake_embed, response, cap=10)
    seed(f, 3)
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["created_preferences"] == 1 and stats["created_schemas"] == 0
    (pref,) = f.vector.get_many({"layer": Layer.L3_DERIVED.value})
    assert pref.kind == "preference" and pref.category.value == "preference"
    assert sorted(pref.custom["source_ids"]) == ["0", "1", "2"]
    f.close()


async def test_duplicate_derived_memory_is_skipped(tmp_storage, fake_embed):
    def response(system, user):
        ids = [x["id"] for x in json.loads(user)["facts"]]
        return {
            "memories": [{"content": "same inference", "kind": "schema", "evidence_ids": ids}],
            "processed_ids": ids,
        }

    f = factory(tmp_storage, fake_embed, response, cap=1)
    seed(f, 2)
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["created_schemas"] == 1 and stats["duplicates_skipped"] == 1
    assert len(f.vector.get_many({"layer": Layer.L3_DERIVED.value})) == 1
    f.close()


async def test_no_inference_marks_facts_processed(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed, {"memories": [], "processed_ids": ["0"]})
    seed(f, 1)
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert f.jev.calls == [] and f.vector.get("0").s2_evidence_count == 1
    assert f.vector.get_many({"layer": Layer.L3_DERIVED.value}) == []
    f.close()


async def test_graph_failure_repaired_after_evidence_watermark_commit(
    tmp_storage, fake_embed, monkeypatch
):
    f = factory(tmp_storage, fake_embed)
    seed(f, 1)
    original = f.graph.add_evidence
    fail = True

    def add_evidence(**kw):
        nonlocal fail
        if fail:
            fail = False
            raise OSError("graph temporarily unavailable")
        return original(**kw)

    monkeypatch.setattr(f.graph, "add_evidence", add_evidence)
    with pytest.raises(OSError):
        await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert f.vector.get("0").s2_evidence_count == 1
    assert len(f.vector.pending_projections(("app", "u", ""))) == 1
    calls = len(f.llm.calls)
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert len(f.llm.calls) == calls
    derived = f.vector.get_many({"layer": Layer.L3_DERIVED.value})[0]
    assert f.graph.evidence_of(derived.node_id) == ["0"]
    assert f.vector.pending_projections(("app", "u", "")) == []
    f.close()


def test_topic_order_keeps_every_fact_and_groups_neighbours():
    from dual_mem.system2.system2_agent import topic_order

    def node(nid, vec, at):
        return MemoryNode(
            node_id=nid, content=nid, layer=Layer.L1_FACT, app_id="a", user_id="u",
            embedding=vec, memory_at=at,
        )

    facts = [
        node("travel1", [1.0, 0.0], 1),
        node("food1", [0.0, 1.0], 2),
        node("travel2", [0.99, 0.05], 3),
        node("food2", [0.05, 0.99], 4),
        node("noembed", None, 5),
    ]
    ordered = [f.node_id for f in topic_order(facts)]
    assert sorted(ordered) == sorted(f.node_id for f in facts)
    assert ordered == ["travel1", "travel2", "food2", "food1", "noembed"]


async def test_compact_evidence_ids_are_mapped_back_and_time_is_not_generation_time(tmp_storage, fake_embed):
    def response(system, user):
        facts = json.loads(user)["facts"]
        assert {f["id"] for f in facts} == {"f0", "f1"}
        assert all(f["conversation_date"] for f in facts)
        return {"memories": [{"content": "User completed two electronics projects", "kind": "summary",
                               "evidence_ids": [f["id"] for f in facts]}]}
    f = factory(tmp_storage, fake_embed, response, cap=2)
    originals = [MemoryNode(node_id=f"long-original-id-{i}", content=f"project {i}",
                            layer=Layer.L1_FACT, app_id="app", user_id="u", memory_at=100 + i * 100,
                            embedding=fake_embed.embed_sync(f"project {i}")) for i in range(2)]
    f.vector.upsert(originals)
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    (derived,) = f.vector.get_many({"layer": Layer.L3_DERIVED.value})
    assert set(derived.custom["source_ids"]) == {n.node_id for n in originals}
    assert derived.memory_at == 200
    assert derived.gmt_created > 200
    assert all(len(call["state"]["facts"]) == 2 for call in f.jev.calls)
    f.close()


async def test_cognition_uses_superseded_reasons_without_promoting_old_values(tmp_storage, fake_embed):
    def response(system, user):
        body = json.loads(user)
        (head,) = body["facts"]
        assert "SUPERSEDED" in system
        assert head["history"][0]["is_latest"] is False
        assert "convenience" in head["history"][0]["content"]
        assert "id" not in head["history"][0]
        return {"memories": [{"content": "User moved from convenience to privacy-driven tool choices",
                               "kind": "summary", "evidence_ids": [head["id"]]}]}
    f = factory(tmp_storage, fake_embed, response)
    old = MemoryNode(node_id="old", content="Used a cloud notebook for convenience",
                     layer=Layer.L2_PROFILE, app_id="app", user_id="u", is_latest=False,
                     memory_at=100, embedding=fake_embed.embed_sync("convenience"))
    head = MemoryNode(node_id="head", content="Now uses local notes because of privacy requirements",
                      layer=Layer.L2_PROFILE, app_id="app", user_id="u", supersedes=["old"],
                      memory_at=200, embedding=fake_embed.embed_sync("privacy"))
    f.vector.upsert([old, head])
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    (derived,) = f.vector.get_many({"layer": Layer.L3_DERIVED.value})
    assert set(derived.custom["source_ids"]) == {"head", "old"}
    assert f.vector.get("head").s2_evidence_count == 1
    assert f.vector.get("old").s2_evidence_count == 0
    assert not f.vector.get("old").is_latest
    assert f.jev.calls[0]["state"]["facts"][0]["history"]
    assert derived.memory_at == 200
    f.close()


async def test_history_cannot_cross_scope(tmp_storage, fake_embed):
    from dual_mem.errors import DecisionContextError
    f = factory(tmp_storage, fake_embed)
    nodes = [MemoryNode(node_id="head", content="current", layer=Layer.L1_FACT,
                        app_id="app", user_id="u", supersedes=["foreign"],
                        embedding=fake_embed.embed_sync("current")),
             MemoryNode(node_id="foreign", content="private", layer=Layer.L1_FACT,
                        app_id="app", user_id="other", is_latest=False,
                        embedding=fake_embed.embed_sync("private"))]
    f.vector.upsert(nodes)
    with pytest.raises(DecisionContextError, match="crosses tenant"):
        await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert f.llm.calls == []
    f.close()


async def test_return_to_old_derived_content_has_new_evidence_identity(tmp_storage, fake_embed):
    import uuid
    text = "User seems to prefer quiet venues"
    def response(system, user):
        ids = [f["id"] for f in json.loads(user)["facts"]]
        return {"memories": [{"content": text, "kind": "preference", "evidence_ids": ids}]}
    f = factory(tmp_storage, fake_embed, response)
    seed(f, 1)
    old_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"s2:app:u::{text}"))
    f.vector.upsert([MemoryNode(node_id=old_id, content=text, layer=Layer.L3_DERIVED,
                               kind="preference", app_id="app", user_id="u", is_latest=False,
                               embedding=fake_embed.embed_sync(text))])
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["created_preferences"] == 1 and stats["failed_batches"] == 0
    derived = [n for n in f.vector.get_many({"layer": Layer.L3_DERIVED.value}) if n.is_latest]
    assert len(derived) == 1 and derived[0].node_id != old_id
    f.close()
