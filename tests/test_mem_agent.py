import pytest
from conftest import FakeJev, FakeLLMClient
from dual_mem import MemoryClient
from dual_mem.agent.reconciler import DecisionContextError
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryStatus


def client(tmp_storage, embed, extract, jev=None, mode="system1", **kw):
    return MemoryClient(
        settings=Settings(storage_dir=tmp_storage, mode=mode, **kw),
        embed=embed,
        llm=FakeLLMClient({"extract": extract}),
        jev=jev or FakeJev(),
    )


async def test_one_generation_and_two_jev_phases_for_facts_profiles_intentions(
    tmp_storage, fake_embed
):
    def script(state, questions):
        return {
            "p0:representation": "FACT",
            "p1:representation": "PROFILE",
            "p2:representation": "INTENTION",
        }

    jev = FakeJev(script)
    c = client(
        tmp_storage,
        fake_embed,
        {
            "facts": [{"content": "Paid 80 yuan"}],
            "identity": [{"content": "Lives in Beijing"}],
            "intentions": [{"content": "Will travel tomorrow"}],
        },
        jev,
    )
    result = await c.add(
        content="Paid 80 yuan, live in Beijing and will travel", user_id="u"
    )
    nodes = c.factory.vector.get_by_ids(result.extra_node_ids)
    assert {n.layer for n in nodes.values()} == {Layer.L1_FACT, Layer.L2_PROFILE}
    assert {n.kind for n in nodes.values()} == {"fact", "profile", "intention"}
    assert len(c.factory.llm.calls) == 1 and len(jev.calls) == 2
    assert all(
        (n.custom or {})["source_ids"] == [result.memory_id] for n in nodes.values()
    )
    assert c.factory.vector.get(result.memory_id).layer == Layer.L0_RAW


@pytest.mark.parametrize("mode", ["system1", "dual"])
async def test_extractor_ephemeral_field_cannot_discard_update(
    tmp_storage, fake_embed, mode
):
    c = client(
        tmp_storage,
        fake_embed,
        {"facts": [{"content": "Moved to Shanghai"}], "is_ephemeral": True},
        mode=mode,
    )
    result = await c.add(content="Moved to Shanghai", user_id="u")
    assert (
        result.commit_passed and result.extracted_count == 1 and not result.is_ephemeral
    )
    assert c.factory.vector.get(result.memory_id).status == MemoryStatus.ACTIVE


async def test_empty_extraction_jev_decides_raw_only(tmp_storage, fake_embed):
    c = client(tmp_storage, fake_embed, {"facts": []})
    result = await c.add(content="Thanks", user_id="u")
    assert result.is_ephemeral and not result.commit_passed
    assert len(c.factory.jev.calls) == 1 and c.factory.vector.get(result.memory_id)


async def test_failed_jev_keeps_raw_evidence_for_idempotent_retry(
    tmp_storage, fake_embed
):
    jev = FakeJev(RuntimeError("service unavailable"))
    c = client(tmp_storage, fake_embed, {"facts": [{"content": "Moved"}]}, jev)
    with pytest.raises(RuntimeError, match="unavailable"):
        await c.add(content="Moved", user_id="u", request_id="retry-me")
    assert len(c.factory.vector.get_many({"user_id": "u"}, limit=100)) == 1
    jev.responses = None
    result = await c.add(content="Moved", user_id="u", request_id="retry-me")
    assert result.extracted_count == 1
    repeated = await c.add(content="Moved", user_id="u", request_id="retry-me")
    assert repeated.extra_node_ids == result.extra_node_ids
    assert len(c.factory.vector.get_many({"user_id": "u"}, limit=100)) == 2
    assert len(jev.calls) == 2
    with pytest.raises(ValueError, match="different input"):
        await c.add(content="different", user_id="u", request_id="retry-me")


async def test_exact_text_distinct_events_are_not_hash_deduplicated(
    tmp_storage, fake_embed
):
    c = client(
        tmp_storage, fake_embed, {"facts": [{"content": "Bought coffee for 20 yuan"}]}
    )
    first = await c.add(content="Bought coffee for 20 yuan", user_id="u", memory_at=100)
    second = await c.add(
        content="Bought coffee for 20 yuan", user_id="u", memory_at=200
    )
    assert (
        first.memory_id != second.memory_id
        and first.extra_node_ids != second.extra_node_ids
    )
    assert len(c.factory.jev.calls) == 3


async def test_batch_skip_predecessor_resolves_to_current_head(tmp_storage, fake_embed):
    c = client(tmp_storage, fake_embed, {"facts": [{"content": "Old"}]})
    result = await c.add(content="Old", user_id="u")
    old = result.extra_node_ids[0]
    c.factory.llm.responses = {
        "extract": {"facts": [{"content": "Old"}, {"content": "New"}]}
    }
    c.factory.jev.responses = {
        "p0:action": "SKIP",
        "p0:target": old,
        "p1:action": "ADD_OVERRIDE",
        "p1:target": "p0",
    }
    updated = await c.add(content="Old to New", user_id="u")
    assert c.factory.vector.get(updated.extra_node_ids[0]).supersedes == [old]
    assert not c.factory.vector.get(old).is_latest


async def test_model_missed_durable_extraction_fails_without_silent_drop(
    tmp_storage, fake_embed
):
    c = client(tmp_storage, fake_embed, {"facts": []}, FakeJev({"value": "REEXTRACT"}))
    with pytest.raises(DecisionContextError, match="missed"):
        await c.add(content="Moved to Beijing", user_id="u")
    assert len(c.factory.vector.get_many({"user_id": "u"}, limit=100)) == 1


async def test_request_retry_after_result_cache_failure_does_not_rerun_jev(
    tmp_storage, fake_embed, monkeypatch
):
    c = client(tmp_storage, fake_embed, {"facts": [{"content": "Fact"}]})
    original = c.factory.cache.set_profile

    def fail(*args, **kwargs):
        raise OSError("cache failed after commit")

    monkeypatch.setattr(c.factory.cache, "set_profile", fail)
    with pytest.raises(OSError):
        await c.add(content="Fact", user_id="u", request_id="durable")
    calls = len(c.factory.jev.calls)
    monkeypatch.setattr(c.factory.cache, "set_profile", original)
    retried = await c.add(content="Fact", user_id="u", request_id="durable")
    assert retried.extracted_count == 1 and len(c.factory.jev.calls) == calls
    assert len(c.factory.vector.get_many({"user_id": "u"}, limit=100)) == 2


async def test_partial_extraction_coverage_cannot_silently_drop_update(
    tmp_storage, fake_embed
):
    from conftest import FakeLLMClient, FakeJev
    from dual_mem import MemoryClient
    from dual_mem.config import Settings
    from dual_mem.agent.reconciler import DecisionContextError

    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient({"extract": {"facts": [{"content": "User is named Alice"}]}}),
        jev=FakeJev({"coverage": "INVALID"}),
    )
    with pytest.raises(DecisionContextError, match="coverage"):
        await c.add(content="My name is Alice and I moved to Shanghai", user_id="u")
    raw = c.factory.vector.get_many({"layer": "L0_RAW"})
    assert len(raw) == 1 and "Shanghai" in raw[0].content
    assert c.factory.vector.get_many({"layer": "L1_FACT"}) == []
    await c.aclose()


async def test_conflicting_batch_targets_replan_with_jev_feedback(
    tmp_storage, fake_embed
):
    from conftest import FakeLLMClient, FakeJev
    from dual_mem import MemoryClient
    from dual_mem.config import Settings
    from dual_mem.types import MemoryNode, Layer

    def script(state, questions):
        ids = [n["id"] for n in state["current"]]
        fixed = bool(state.get("previous_plan_rejected"))
        return {
            "p0:action": "ADD_SUPPLEMENT" if fixed else "ADD_OVERRIDE",
            "p0:target": "NONE" if fixed else ids[0],
            "p1:action": "ADD_OVERRIDE",
            "p1:target": ids[0],
        }

    j = FakeJev(script)
    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(
            {
                "extract": {
                    "facts": [
                        {"content": "Moved Beijing to Shanghai"},
                        {"content": "Currently Shanghai"},
                    ]
                }
            }
        ),
        jev=j,
    )
    c.factory.vector.upsert(
        [
            MemoryNode(
                node_id="old",
                content="Currently Beijing",
                layer=Layer.L2_PROFILE,
                app_id="default",
                user_id="u",
                embedding=fake_embed.embed_sync("Currently Beijing"),
            )
        ]
    )
    r = await c.add(content="Moved Beijing to Shanghai, now Shanghai", user_id="u")
    assert len(j.calls) == 4 and j.calls[2]["state"]["previous_plan_rejected"]
    assert len(r.extra_node_ids) == 2 and not c.factory.vector.get("old").is_latest
    assert len(c.factory.llm.calls) == 1
    await c.aclose()


async def test_target_none_constrains_action_then_writes(
    tmp_storage, fake_embed
):
    def script(state, questions):
        return {"p1:action": "ADD_SUPPLEMENT", "p1:target": "NONE"}

    jev = FakeJev(script)
    c = client(
        tmp_storage,
        fake_embed,
        {
            "facts": [
                {"content": "用户喜欢喝茶"},
                {"content": "用户住在北京"},
            ]
        },
        jev=jev,
    )
    result = await c.add(content="我喜欢喝茶，住在北京", user_id="u")
    assert result.extracted_count == 2
    action_options = jev.calls[1]["questions"]["p1:action"]["criteria"]
    assert "ADD_SUPPLEMENT" in action_options and "SKIP" not in action_options
    assert all(
        not any(name.endswith(":repair") for name in call["questions"])
        for call in jev.calls
    )
    await c.aclose()


async def test_relation_to_raw_only_proposal_replans_batch(tmp_storage, fake_embed):
    def script(state, questions):
        fixed = bool(state.get("previous_plan_rejected"))
        return {
            "p0:action": "RAW_ONLY",
            "p1:target": "NONE" if fixed else "p0",
            "p1:action": "ADD_SUPPLEMENT" if fixed else "SKIP",
        }

    jev = FakeJev(script)
    c = client(
        tmp_storage,
        fake_embed,
        {
            "facts": [
                {"content": "非持久化候选"},
                {"content": "应独立保存的事实"},
            ]
        },
        jev=jev,
    )
    result = await c.add(content="包含两个候选", user_id="u")
    assert result.extracted_count == 1
    assert len(jev.calls) == 4
    feedback = jev.calls[2]["state"]["previous_plan_rejected"]
    assert feedback["decisions"][0]["action"] == "RAW_ONLY"
    await c.aclose()
