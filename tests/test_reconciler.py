import json

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
    "relation,target,links,conflicts",
    [
        ("OVERRIDE", "old", ["old"], []),
        ("NEGATE", "old", ["old"], []),
        ("TEMPORAL", "NONE", [], []),
        ("SUPPLEMENT", "NONE", [], []),
        ("CONFLICT", "old", [], ["old"]),
    ],
)
async def test_model_selects_five_relations_without_semantic_repair(
    store, fake_embed, relation, target, links, conflicts
):
    seed(store, fake_embed)
    jev = FakeJev({"p0:action": "ADD_" + relation, "p0:target": target})
    (op,) = await decide(store, fake_embed, jev)
    assert op.update_type == relation and op.supersedes == links
    assert op.conflicts_with == conflicts
    assert op.target_ids == ([] if target == "NONE" else ["old"])
    assert len(jev.calls) == 2


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
    assert op.op == "RAW_ONLY" and op.memory_id is None


async def test_need_context_appends_independently_for_review(store, fake_embed):
    seed(store, fake_embed)
    (op,) = await decide(
        store,
        fake_embed,
        FakeJev({"p0:action": "NEED_CONTEXT", "p0:target": "old"}),
    )
    assert op.op == "ADD" and op.update_type == "SUPPLEMENT"
    assert op.supersedes == [] and op.target_ids == []
    assert op.needs_review and op.review_reason == "NEED_CONTEXT"
    assert op.review_targets == ["old"]
    assert op.decision["action"]["choice"] == "NEED_CONTEXT"
    assert op.decision["effective"] == "ADD_SUPPLEMENT"


async def test_invalid_candidate_is_rejected_not_written(store, fake_embed):
    seed(store, fake_embed)
    (op,) = await decide(
        store, fake_embed, FakeJev({"p0:action": "INVALID", "p0:target": "old"})
    )
    assert op.op == "INVALID" and not op.supersedes and not op.needs_review


async def test_low_confidence_relation_is_deferred(store, fake_embed):
    from dual_mem.providers.jev import JevAnswer

    seed(store, fake_embed)
    weak = JevAnswer(
        type="choice",
        choice="ADD_OVERRIDE",
        confidence=0.4,
        probabilities={"ADD_OVERRIDE": 0.4, "ADD_MERGE": 0.35},
    )
    jev = FakeJev({"p0:action": weak, "p0:target": "old"})
    (op,) = await decide(store, fake_embed, jev, review_confidence=0.6)
    assert op.update_type == "SUPPLEMENT" and op.supersedes == []
    assert op.review_reason == "low-confidence ADD_OVERRIDE"
    (strong,) = await decide(store, fake_embed, jev, review_confidence=0.3)
    assert strong.update_type == "OVERRIDE" and strong.supersedes == ["old"]


async def test_unsure_multiple_targets_keep_decision_and_defer(store, fake_embed):
    seed(store, fake_embed, nid="a", content="Part A")
    seed(store, fake_embed, nid="b", content="Part B")

    def script(state, questions):
        if "p0:target" in questions:
            return {"p0:target": "MULTIPLE"}
        if "p0:action" in questions:
            return {"p0:action": "ADD_SUPPLEMENT"}
        return {"p0:a": "INCLUDE", "p0:b": "UNSURE"}

    (op,) = await decide(store, fake_embed, FakeJev(script))
    assert op.update_type == "SUPPLEMENT" and op.target_ids == []
    assert op.review_reason == "uncertain multiple targets"
    assert set(op.review_targets) == {"a", "b"}


async def test_older_source_cannot_override_newer_head(store, fake_embed):
    seed(store, fake_embed, memory_at=2_000)
    jev = FakeJev({"p0:action": "ADD_HISTORICAL", "p0:target": "old"})
    rec = Reconciler(jev=jev, llm=FakeLLMClient(), embed=fake_embed, vector=store)
    (op,) = await rec.reconcile(
        new_memories=["User lives in Beijing"],
        new_memories_meta=[{}],
        app_id="app",
        user_id="u",
        agent_id="",
        current_time="2026-10-15",
        memory_at=1_000,
    )
    criteria = jev.calls[1]["questions"]["p0:action"]["criteria"]
    assert "ADD_HISTORICAL" in criteria
    assert not {"ADD_OVERRIDE", "ADD_NEGATE", "DELETE"} & set(criteria)
    assert op.update_type == "HISTORICAL" and op.target_ids == ["old"]
    assert op.supersedes == []


async def test_newer_source_is_not_offered_historical(store, fake_embed):
    seed(store, fake_embed, memory_at=1_000)
    jev = FakeJev({"p0:action": "ADD_OVERRIDE", "p0:target": "old"})
    rec = Reconciler(jev=jev, llm=FakeLLMClient(), embed=fake_embed, vector=store)
    await rec.reconcile(
        new_memories=["User lives in Beijing"],
        new_memories_meta=[{}],
        app_id="app",
        user_id="u",
        agent_id="",
        current_time="2026-10-15",
        memory_at=2_000,
    )
    criteria = jev.calls[1]["questions"]["p0:action"]["criteria"]
    assert "ADD_OVERRIDE" in criteria and "ADD_HISTORICAL" not in criteria


async def test_target_none_excludes_actions_that_require_target(store, fake_embed):
    seed(store, fake_embed)
    jev = FakeJev({"p0:action": "ADD_SUPPLEMENT", "p0:target": "NONE"})
    (op,) = await decide(store, fake_embed, jev)
    assert op.update_type == "SUPPLEMENT" and op.supersedes == []
    action_options = jev.calls[1]["questions"]["p0:action"]["criteria"]
    assert set(action_options) == {
        "ADD_SUPPLEMENT",
        "ADD_TEMPORAL",
        "RAW_ONLY",
        "INVALID",
        "NEED_CONTEXT",
    }
    target_instructions = jev.calls[0]["questions"]["p0:target"]["instructions"]
    assert "Choose NONE" in target_instructions and "TEMPORAL" in target_instructions


async def test_selected_target_excludes_append_actions(store, fake_embed):
    seed(store, fake_embed)
    jev = FakeJev({"p0:action": "ADD_OVERRIDE", "p0:target": "old"})
    (op,) = await decide(store, fake_embed, jev)
    assert op.target_ids == ["old"] and op.supersedes == ["old"]
    assert set(jev.calls[1]["questions"]["p0:action"]["criteria"]) == {
        "ADD_OVERRIDE",
        "ADD_MERGE",
        "ADD_NEGATE",
        "ADD_CONFLICT",
        "SKIP",
        "DELETE",
        "INVALID",
        "NEED_CONTEXT",
    }


async def test_injected_provider_must_answer_every_question(store, fake_embed):
    seed(store, fake_embed)

    class IncompleteJev(FakeJev):
        async def evaluate(self, *, state, questions):
            answers = await super().evaluate(state=state, questions=questions)
            answers.pop(next(iter(answers)))
            return answers

    with pytest.raises(DecisionContextError, match="exact question coverage"):
        await decide(
            store,
            fake_embed,
            IncompleteJev({"p0:action": "ADD_OVERRIDE", "p0:target": "old"}),
        )


async def test_multiple_targets_exclude_skip_alias(store, fake_embed):
    seed(store, fake_embed, nid="a", content="Part A")
    seed(store, fake_embed, nid="b", content="Part B")

    def script(state, questions):
        if "p0:action" in questions:
            return {"p0:action": "ADD_CONFLICT"}
        if "p0:target" in questions:
            return {"p0:target": "MULTIPLE"}
        return {name: "INCLUDE" for name in questions}

    jev = FakeJev(script)
    (op,) = await decide(store, fake_embed, jev)
    action_call = next(
        call for call in jev.calls if "p0:action" in call["questions"]
    )
    assert "SKIP" not in action_call["questions"]["p0:action"]["criteria"]
    assert set(op.conflicts_with) == {"a", "b"}


async def test_earlier_proposal_target_excludes_delete(store, fake_embed):
    jev = FakeJev(
        {
            "p0:action": "ADD_SUPPLEMENT",
            "p1:target": "p0",
            "p1:action": "SKIP",
        }
    )
    rec = Reconciler(
        jev=jev,
        llm=FakeLLMClient(),
        embed=fake_embed,
        vector=store,
    )
    await rec.reconcile(
        new_memories=["First", "First"],
        new_memories_meta=[{}, {}],
        app_id="app",
        user_id="u",
        agent_id="",
        current_time="2026-10-15",
    )
    action_options = jev.calls[1]["questions"]["p1:action"]["criteria"]
    assert "SKIP" in action_options and "DELETE" not in action_options


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
    assert len(llm.calls) == 1 and len(jev.calls) == 3


async def test_large_scope_orders_material_by_similarity_without_gating(
    store, fake_embed
):
    for i in range(5):
        seed(store, fake_embed, nid=str(i), content=f"fact {i}")
    jev = FakeJev({"p0:action": "ADD_SUPPLEMENT", "p0:target": "NONE"})
    (op,) = await decide(store, fake_embed, jev, content="fact 3", candidate_limit=2)
    # Routing never decides the write: the proposal is still decided and appended.
    assert op.op == "ADD" and op.update_type == "SUPPLEMENT"
    state = jev.calls[0]["state"]
    # The nearest head (identical text) is always compared; the rest is capped.
    assert len(state["current"]) == 2 and state["current"][0]["id"] == "3"
    assert len(jev.calls) == 2
    # The proposal embedding computed for routing is reused at write time.
    assert op.embedding == fake_embed.embed_sync("fact 3")


async def test_material_budget_keeps_most_similar_heads(store, fake_embed):
    for i in range(6):
        seed(store, fake_embed, nid=str(i), content=f"fact {i} " + "x" * 400)
    jev = FakeJev()
    await decide(
        store,
        fake_embed,
        jev,
        content="fact 4 " + "x" * 400,
        max_state_chars=7_000,
    )
    ids = [c["id"] for c in jev.calls[0]["state"]["current"]]
    assert 0 < len(ids) < 6 and ids[0] == "4"
    assert len(json.dumps(jev.calls[0]["state"], ensure_ascii=False)) < 7_000


async def test_oversized_source_fails_before_any_jev_call(store, fake_embed):
    jev = FakeJev()
    rec = Reconciler(
        jev=jev, llm=FakeLLMClient(), embed=fake_embed, vector=store, max_state_chars=5_000
    )
    with pytest.raises(DecisionContextError, match="split the source"):
        await rec.reconcile(
            new_memories=["x"],
            new_memories_meta=[{}],
            app_id="app",
            user_id="u",
            agent_id="",
            current_time="2026-10-15",
            source_content="y" * 6_000,
        )
    assert jev.calls == []


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
    assert len(llm.calls) == 1 and len(jev.calls) == 3


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


async def test_target_choices_are_short_labels_full_node_in_state(store, fake_embed):
    profile = MemoryNode(
        node_id="profile",
        content="User lives in Shanghai " + "detail " * 80,
        layer=Layer.L2_PROFILE,
        kind="profile",
        app_id="app",
        user_id="u",
        embedding=fake_embed.embed_sync("profile"),
        memory_at=1_792_022_400,
        custom={"source_ids": ["raw"]},
    )
    store.upsert([profile])
    jev = FakeJev({"p0:action": "ADD_OVERRIDE", "p0:target": "profile"})
    await decide(store, fake_embed, jev)
    label = jev.calls[0]["questions"]["p0:target"]["criteria"]["profile"]
    assert label.startswith("[L2_PROFILE/profile 2026-10-15] User lives in Shanghai")
    assert len(label) < 300 and label.endswith("…")
    (node,) = jev.calls[0]["state"]["current"]
    assert node["source_ids"] == ["raw"] and node["content"] == profile.content
