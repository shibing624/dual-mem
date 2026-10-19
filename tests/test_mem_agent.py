import pytest
from conftest import FakeJev, FakeLLMClient
from dual_mem import MemoryClient
from dual_mem.agent.reconciler import DecisionContextError
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryStatus


def _head(embed, content, nid="seed"):
    from dual_mem.types import MemoryNode

    return MemoryNode(
        node_id=nid, content=content, layer=Layer.L1_FACT, app_id="default",
        user_id="u", embedding=embed.embed_sync(content),
    )


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
    jev = FakeJev()
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
    # One generation; each later proposal has its own small target state.
    # NONE everywhere means no action round or retry.
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


async def test_empty_extraction_is_raw_only_without_model_call(tmp_storage, fake_embed):
    c = client(tmp_storage, fake_embed, {"facts": []})
    result = await c.add(content="Thanks", user_id="u")
    assert result.is_ephemeral and not result.commit_passed
    assert len(c.factory.jev.calls) == 0 and c.factory.vector.get(result.memory_id)


async def test_failed_jev_keeps_raw_evidence_for_idempotent_retry(
    tmp_storage, fake_embed
):
    jev = FakeJev(RuntimeError("service unavailable"))
    c = client(tmp_storage, fake_embed, {"facts": [{"content": "Moved"}]}, jev)
    c.factory.vector.upsert([_head(fake_embed, "Lives somewhere")])
    with pytest.raises(RuntimeError, match="unavailable"):
        await c.add(content="Moved", user_id="u", request_id="retry-me")
    assert len(c.factory.vector.get_many({"user_id": "u"}, limit=100)) == 2
    jev.responses = None
    result = await c.add(content="Moved", user_id="u", request_id="retry-me")
    assert result.extracted_count == 1
    repeated = await c.add(content="Moved", user_id="u", request_id="retry-me")
    assert repeated.extra_node_ids == result.extra_node_ids
    assert len(c.factory.vector.get_many({"user_id": "u"}, limit=100)) == 3
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
    # First write has no heads (direct append); the second compares one target.
    assert len(c.factory.jev.calls) == 1


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


async def test_empty_extraction_is_not_regenerated(tmp_storage, fake_embed):
    """One extraction pass only; a miss keeps the source and never re-runs."""
    c = client(tmp_storage, fake_embed, {"facts": []}, FakeJev({"value": "REEXTRACT"}))
    result = await c.add(content="Moved to Beijing", user_id="u")
    extract_calls = [
        call for call in c.factory.llm.calls if "Generate faithful" in call["system"]
    ]
    assert len(extract_calls) == 1
    assert result.status == "raw_only"
    assert result.decisions[0].action == "RAW_ONLY"
    assert len(c.factory.vector.get_many({"user_id": "u"}, limit=100)) == 1


async def test_request_retry_after_result_cache_failure_does_not_rerun_jev(
    tmp_storage, fake_embed, monkeypatch
):
    c = client(tmp_storage, fake_embed, {"facts": [{"content": "Fact"}]})
    original = c.factory.cache.set_request_outcome

    def fail(*args, **kwargs):
        raise OSError("cache failed after commit")

    monkeypatch.setattr(c.factory.cache, "set_request_outcome", fail)
    with pytest.raises(OSError):
        await c.add(content="Fact", user_id="u", request_id="durable")
    calls = len(c.factory.jev.calls)
    monkeypatch.setattr(c.factory.cache, "set_request_outcome", original)
    retried = await c.add(content="Fact", user_id="u", request_id="durable")
    assert retried.extracted_count == 1 and len(c.factory.jev.calls) == calls
    assert len(c.factory.vector.get_many({"user_id": "u"}, limit=100)) == 2


async def test_rejected_coverage_still_commits_candidates(tmp_storage, fake_embed):
    """A coverage rejection is logged; the candidates are decided and committed anyway."""
    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient({"extract": {"facts": [{"content": "User is named Alice"}]}}),
        jev=FakeJev({"coverage": "INVALID"}),
    )
    result = await c.add(
        content="My name is Alice and I moved to Shanghai", user_id="u"
    )
    raw = c.factory.vector.get_many({"layer": "L0_RAW"})
    assert len(raw) == 1 and "Shanghai" in raw[0].content
    (fact,) = c.factory.vector.get_many({"layer": "L1_FACT"})
    assert fact.content == "User is named Alice"
    assert result.status != "raw_only"
    assert all("coverage" not in call["questions"] for call in c.factory.jev.calls)
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
    # The collision is settled inside the batch (no Jev replan): one claim retires the
    # head, the other is written as an independent memory, and both survive.
    assert len(j.calls) == 4 and all(
        "previous_plan_rejected" not in call["state"] for call in j.calls
    )
    assert len(r.extra_node_ids) == 2 and not c.factory.vector.get("old").is_latest
    assert sorted(d.update_type for d in r.decisions) == ["OVERRIDE", "SUPPLEMENT"]
    assert len(c.factory.llm.calls) == 1
    await c.aclose()


async def test_most_confident_claim_wins_the_target(tmp_storage, fake_embed):
    """When two proposals claim one head, the higher-confidence relation is kept."""
    from dual_mem.providers.jev import JevAnswer
    from dual_mem.types import MemoryNode

    def script(state, questions):
        ids = [n["id"] for n in state["current"]]
        return {
            "p0:action": JevAnswer(
                type="choice", choice="ADD_OVERRIDE", confidence=0.6,
                probabilities={"ADD_OVERRIDE": 0.6},
            ),
            "p0:target": ids[0],
            "p1:action": JevAnswer(
                type="choice", choice="ADD_OVERRIDE", confidence=0.9,
                probabilities={"ADD_OVERRIDE": 0.9},
            ),
            "p1:target": ids[0],
        }

    j = FakeJev(script)
    c = client(
        tmp_storage,
        fake_embed,
        {"facts": [{"content": "Moved to Shanghai"}, {"content": "Currently Shanghai"}]},
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
    r = await c.add(content="Moved to Shanghai, now Shanghai", user_id="u")
    assert not c.factory.vector.get("old").is_latest
    # p1 (0.9) keeps the override; p0 (0.6) is appended independently.
    kept = [d for d in r.decisions if d.update_type == "OVERRIDE"]
    assert len(kept) == 1 and kept[0].proposal_ids == ["p1"]
    assert len(r.extra_node_ids) == 2
    await c.aclose()


async def test_stale_cas_raises_without_internal_retry(
    tmp_storage, fake_embed, monkeypatch
):
    """A lost CAS surfaces immediately; the caller owns retrying."""
    from dual_mem import StaleDecisionError
    c = client(tmp_storage, fake_embed, {"facts": [{"content": "Lives in Beijing"}]})
    c.factory.vector.upsert([_head(fake_embed, "Lives in Shanghai")])
    c.factory.jev.responses = {"p0:target": "seed", "p0:action": "ADD_OVERRIDE"}
    store = type(c.factory.vector)
    original = store.apply_version_plan
    calls = {"n": 0}

    def flaky(self, **kw):
        calls["n"] += 1
        return False if calls["n"] <= 1 else original(self, **kw)

    monkeypatch.setattr(store, "apply_version_plan", flaky)
    with pytest.raises(StaleDecisionError):
        await c.add(content="I live in Beijing", user_id="u")
    assert calls["n"] == 1
    # The same call succeeds once the conflict is gone.
    r = await c.add(content="I live in Beijing", user_id="u", request_id="retry-1")
    assert r.status == "committed" and len(r.extra_node_ids) == 1
    assert not c.factory.vector.get("seed").is_latest
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
    # NONE is an independent append: no action question is asked.
    assert len(jev.calls) == 1 and "p1:action" not in jev.calls[0]["questions"]
    assert result.decisions[1].update_type == "SUPPLEMENT"
    assert all(
        not any(name.endswith(":repair") for name in call["questions"])
        for call in jev.calls
    )
    await c.aclose()


async def test_relation_to_raw_only_proposal_is_settled_without_replan(
    tmp_storage, fake_embed
):
    """A dangling relation is dropped in-batch; no second decision round."""
    def script(state, questions):
        return {
            "p0:target": "seed",
            "p0:action": "INVALID",
            "p1:target": "p0",
            "p1:action": "SKIP",
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
    c.factory.vector.upsert([_head(fake_embed, "种子")])
    result = await c.add(content="包含两个候选", user_id="u")
    assert result.extracted_count == 1
    # One decision round: the SKIP that pointed at the RAW_ONLY proposal is written
    # as an independent memory instead of triggering a replan.
    assert len(jev.calls) == 4
    assert result.decisions[1].update_type == "SUPPLEMENT"
    await c.aclose()


async def test_extracted_speakers_are_preserved_and_assistant_is_not_user_profile(tmp_storage, fake_embed):
    c = client(tmp_storage, fake_embed, {"facts": [
        {"content": "The user owns a telemetry watch", "type": "profile", "speaker": "user"},
        {"content": "The assistant recommended a monitoring service", "type": "profile", "speaker": "assistant"},
    ]})
    result = await c.add(content="My watch records telemetry. You recommended a monitoring service.", user_id="u")
    nodes = c.factory.vector.get_by_ids(result.extra_node_ids)
    user = next(n for n in nodes.values() if n.owner == "user")
    assistant = next(n for n in nodes.values() if n.owner == "agent")
    assert user.layer == Layer.L2_PROFILE and user.kind == "profile"
    assert assistant.layer == Layer.L1_FACT and assistant.kind == "fact"
    await c.aclose()


async def test_atomic_attribute_value_profiles_are_not_discarded(tmp_storage, fake_embed):
    c = client(tmp_storage, fake_embed, {"identity": [
        {"attribute": "diet", "value": "vegetarian"},
        {"attribute": "residence", "value": "Bristol"},
        {"attribute": "devices", "value": ["ThinkPad T14"]},
    ]})
    result = await c.add(content="Vegetarian, live in Bristol, own a ThinkPad T14", user_id="u")
    nodes = c.factory.vector.get_by_ids(result.extra_node_ids)
    assert len(nodes) == 3 and all(n.layer == Layer.L2_PROFILE for n in nodes.values())
    assert any("vegetarian" in n.content for n in nodes.values())
    assert any("Bristol" in n.content for n in nodes.values())
    assert any("ThinkPad T14" in n.content for n in nodes.values())
    await c.aclose()
