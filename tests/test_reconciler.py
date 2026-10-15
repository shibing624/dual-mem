import pytest
from conftest import FakeJev, FakeLLMClient
from dual_mem.agent.reconciler import Reconciler, DecisionContextError
from dual_mem.storage.vector_store import ChromaVectorStore
from dual_mem.types import Layer, MemoryNode


@pytest.fixture
def store(tmp_storage):
    return ChromaVectorStore(tmp_storage)


def seed(store, fake_embed, nid="old", content="User lives in Shanghai", **kw):
    node = MemoryNode(
        node_id=nid,
        content=content,
        layer=Layer.L1_FACT,
        app_id="app",
        user_id="u",
        embedding=fake_embed.embed_sync(content),
        **kw,
    )
    store.upsert([node])
    return node


async def decide(
    store, embed, jev, content="User lives in Beijing again", llm=None, **kw
):
    rec = Reconciler(
        jev=jev, llm=llm or FakeLLMClient(), embed=embed, vector=store, **kw
    )
    ops = await rec.reconcile(
        new_memories=[content],
        new_memories_meta=[{"content": content}],
        app_id="app",
        user_id="u",
        agent_id="",
        current_time="2026-10-15",
    )
    return ops


@pytest.mark.parametrize(
    "relation,links,conflicts",
    [
        ("OVERRIDE", ["old"], []),
        ("NEGATE", ["old"], []),
        ("TEMPORAL", [], []),
        ("SUPPLEMENT", [], []),
        ("CONFLICT", [], ["old"]),
    ],
)
async def test_model_selects_five_relations_without_semantic_repair(
    store, fake_embed, relation, links, conflicts
):
    seed(store, fake_embed)
    jev = FakeJev({"p0:action": "ADD_" + relation, "p0:target": "old"})
    (op,) = await decide(store, fake_embed, jev)
    assert op.update_type == relation and op.supersedes == links
    assert op.conflicts_with == conflicts
    assert len(jev.calls) == 1


async def test_return_to_old_value_is_update_not_known(store, fake_embed):
    old = seed(
        store,
        fake_embed,
        nid="beijing",
        content="User lives in Beijing",
        is_latest=False,
        superseded_by=["old"],
    )
    seed(store, fake_embed, supersedes=[old.node_id])
    jev = FakeJev({"p0:action": "ADD_OVERRIDE", "p0:target": "old"})
    (op,) = await decide(store, fake_embed, jev)
    assert op.supersedes == ["old"]
    state = jev.calls[0]["state"]
    assert state["current"][0]["history_versions"][0]["id"] == "beijing"
    assert "beijing" not in jev.calls[0]["questions"]["p0:target"]["criteria"]


async def test_known_and_no_durable_value_are_distinct_model_decisions(
    store, fake_embed
):
    seed(store, fake_embed)
    (op,) = await decide(
        store, fake_embed, FakeJev({"p0:action": "SKIP", "p0:target": "old"})
    )
    assert op.memory_id == "old"
    (op,) = await decide(
        store, fake_embed, FakeJev({"p0:action": "RAW_ONLY", "p0:target": "NONE"})
    )
    assert op.op == "SKIP" and op.decision["action"]["choice"] == "RAW_ONLY"


@pytest.mark.parametrize(
    "answers",
    [
        {"p0:action": "ADD_OVERRIDE", "p0:target": "NONE"},
        {"p0:action": "SKIP", "p0:target": "NONE"},
        {"p0:action": "NEED_CONTEXT", "p0:target": "old"},
    ],
)
async def test_inconsistent_or_incomplete_decision_fails_closed(
    store, fake_embed, answers
):
    seed(store, fake_embed)
    with pytest.raises(DecisionContextError):
        await decide(store, fake_embed, FakeJev(answers))


@pytest.mark.parametrize("verdict", ["VALID", "INVALID"])
async def test_merge_has_one_generation_and_losslessness_check(
    store, fake_embed, verdict
):
    seed(store, fake_embed, content="Paid 80 yuan on Monday")
    jev = FakeJev({"p0:action": "ADD_MERGE", "p0:target": "old", "p0": verdict})
    llm = FakeLLMClient(
        {
            "json": {
                "merges": [
                    {
                        "proposal_id": "p0",
                        "content": "Paid 80 yuan on Monday and 30 yuan on Tuesday",
                    }
                ]
            }
        }
    )
    if verdict == "INVALID":
        with pytest.raises(DecisionContextError, match="lossy"):
            await decide(
                store, fake_embed, jev, content="Paid 30 yuan on Tuesday", llm=llm
            )
    else:
        (op,) = await decide(
            store, fake_embed, jev, content="Paid 30 yuan on Tuesday", llm=llm
        )
        assert "80" in op.content and "30" in op.content
    assert len(llm.calls) == 1 and len(jev.calls) == 2


async def test_jev_routes_large_materials_not_embedding_distance(store, fake_embed):
    for i in range(3):
        seed(store, fake_embed, nid=str(i), content=f"fact {i}")

    def script(state, questions):
        if "proposals" in state and "current" in state and "p0:action" not in questions:
            return {q: "INCLUDE" if q == "1" else "EXCLUDE" for q in questions}
        return {"p0:action": "ADD_OVERRIDE", "p0:target": "1"}

    jev = FakeJev(script)
    (op,) = await decide(store, fake_embed, jev, candidate_limit=1)
    assert op.supersedes == ["1"] and len(jev.calls) == 4


async def test_foreign_or_missing_history_is_rejected(store, fake_embed):
    seed(store, fake_embed, supersedes=["missing"])
    with pytest.raises(DecisionContextError, match="missing history"):
        await decide(store, fake_embed, FakeJev())


async def test_dependent_merges_include_transitive_content_and_sources(
    store, fake_embed
):
    seed(
        store, fake_embed, content="Monday paid 80", custom={"source_ids": ["raw-old"]}
    )
    jev = FakeJev(
        {
            "p0:action": "ADD_MERGE",
            "p0:target": "old",
            "p1:action": "ADD_MERGE",
            "p1:target": "p0",
        }
    )
    llm = FakeLLMClient(
        {
            "json": {
                "merges": [
                    {"proposal_id": "p0", "content": "Monday 80 Tuesday 30"},
                    {
                        "proposal_id": "p1",
                        "content": "Monday 80 Tuesday 30 Wednesday 20",
                    },
                ]
            }
        }
    )
    rec = Reconciler(jev=jev, llm=llm, embed=fake_embed, vector=store)
    ops = await rec.reconcile(
        new_memories=["Tuesday paid 30", "Wednesday paid 20"],
        new_memories_meta=[
            {"content": "Tuesday paid 30", "source_ids": ["raw-new"]},
            {"content": "Wednesday paid 20", "source_ids": ["raw-new"]},
        ],
        app_id="app",
        user_id="u",
        agent_id="",
        current_time="2026-10-15",
    )
    import json

    material = json.loads(llm.calls[0]["user"])
    assert material[1]["old"] == ["Tuesday paid 30", "Monday paid 80"]
    assert ops[1].source_ids == ["raw-new", "raw-old"]
    assert len(llm.calls) == 1 and len(jev.calls) == 2


async def test_model_validity_iso_normalized_to_epoch(store, fake_embed):
    r = Reconciler(jev=FakeJev(), llm=FakeLLMClient(), embed=fake_embed, vector=store)
    ops = await r.reconcile(
        new_memories=["Lives Beijing"],
        new_memories_meta=[{"valid_from": "2026-10-15T00:00:00Z"}],
        app_id="app",
        user_id="u",
        agent_id="",
        current_time="2026-10-15",
    )
    assert ops[0].valid_from == 1792022400
    with pytest.raises(DecisionContextError, match="validity"):
        await r.reconcile(
            new_memories=["Lives Beijing"],
            new_memories_meta=[{"valid_from": "yesterday"}],
            app_id="app",
            user_id="u",
            agent_id="",
            current_time="2026-10-15",
        )
