"""End-to-end protocol behaviour added for benchmark-grade ingestion."""

import pytest
from fastapi.testclient import TestClient

from conftest import FakeJev, FakeLLMClient
from dual_mem import DecisionContextError, MemoryClient, StaleDecisionError
from dual_mem.api import create_app
from dual_mem.config import Settings
from dual_mem.providers.jev import JevAnswer
from dual_mem.retrieval.formatter import format_memories, format_search_hit_line
from dual_mem.types import Layer, MemoryNode


def make(tmp_storage, embed, extract, jev=None, **kw):
    return MemoryClient(
        settings=Settings(storage_dir=tmp_storage, **kw),
        embed=embed,
        llm=FakeLLMClient({"extract": extract or echo}),
        jev=jev or FakeJev(),
    )


def echo(system, user):
    """Extract the conversation text verbatim as one candidate."""
    return {"facts": [{"content": user.split("Conversation:\n", 1)[-1].strip()}]}


def head(c, user="u", layer=Layer.L2_PROFILE):
    nodes = c.factory.vector.get_many({"user_id": user, "layer": layer.value}, limit=50)
    return [n for n in nodes if n.is_latest]


async def test_out_of_order_old_session_is_inserted_as_history(tmp_storage, fake_embed):
    def script(state, questions):
        heads = state.get("current", [])
        if "p0:action" in questions and heads:
            choices = questions["p0:action"]["criteria"]
            action = "ADD_HISTORICAL" if "ADD_HISTORICAL" in choices else "ADD_OVERRIDE"
            return {"p0:action": action, "p0:representation": "PROFILE"}
        return {
            "p0:target": heads[0]["id"] if heads else "NONE",
            "p0:representation": "PROFILE",
        }

    c = make(tmp_storage, fake_embed, None, FakeJev(script))
    shanghai = await c.add(content="I live in Shanghai", user_id="u", memory_at=2_000)
    beijing = await c.add(content="I live in Beijing", user_id="u", memory_at=1_000)
    assert beijing.decisions[0].update_type == "HISTORICAL"
    (current,) = head(c)
    assert current.node_id == shanghai.extra_node_ids[0]
    old = c.factory.vector.get(beijing.extra_node_ids[0])
    assert not old.is_latest and old.superseded_by == [current.node_id]
    assert current.supersedes == [old.node_id]
    searched = await c.search(query="where does the user live", user_id="u", min_score=0)
    (item,) = [i for i in searched.memories.flatten() if i.layer == "L2_PROFILE"]
    assert item.content == "I live in Shanghai"
    assert [p.content for p in item.previous] == ["I live in Beijing"]
    await c.aclose()


async def test_historical_cannot_attach_twice_to_one_head_in_a_plan(tmp_storage, fake_embed):
    c = make(tmp_storage, fake_embed, None)
    store = c.factory.vector
    newer = MemoryNode(
        node_id="h", content="now", layer=Layer.L2_PROFILE, app_id="default",
        user_id="u", embedding=fake_embed.embed_sync("now"), memory_at=10,
    )
    store.upsert([newer])
    old_a, old_b = (
        MemoryNode(
            node_id=nid, content=nid, layer=Layer.L2_PROFILE, app_id="default",
            user_id="u", embedding=fake_embed.embed_sync(nid), is_latest=False,
            superseded_by=["h"],
        )
        for nid in ("a", "b")
    )
    head_rev = store.get("h").revision
    assert store.apply_version_plan(
        nodes=[old_a], expected={"h": head_rev}, archives=[], scope=("default", "u", "")
    )
    with pytest.raises(ValueError, match="historical version requires"):
        store.apply_version_plan(
            nodes=[MemoryNode(
                node_id="c", content="c", layer=Layer.L2_PROFILE, app_id="default",
                user_id="u", embedding=fake_embed.embed_sync("c"), is_latest=True,
                superseded_by=["h"],
            )],
            expected={"h": store.get("h").revision},
            archives=[],
            scope=("default", "u", ""),
        )
    assert store.apply_version_plan(
        nodes=[old_b], expected={"h": store.get("h").revision}, archives=[],
        scope=("default", "u", ""),
    )
    assert store.get("h").supersedes == ["a", "b"] and store.get("h").is_latest
    await c.aclose()


async def test_need_context_is_written_then_resolved_by_digest(tmp_storage, fake_embed):
    phase = {"review": False}

    def script(state, questions):
        heads = state.get("current", [])
        if not heads:
            return {}
        if "p0:target" in questions:
            return {"p0:target": heads[0]["id"]}
        if phase["review"]:
            return {"p0:action": "ADD_OVERRIDE"}
        return {"p0:action": "NEED_CONTEXT"}

    c = make(tmp_storage, fake_embed, None, FakeJev(script))
    first = await c.add(content="Lives in Shanghai", user_id="u")
    second = await c.add(content="Lives in Beijing now", user_id="u")
    assert second.status == "pending_review" and second.commit_passed
    assert second.decisions[0].review_reason == "NEED_CONTEXT"
    provisional = c.factory.vector.get(second.extra_node_ids[0])
    assert provisional.is_latest and provisional.custom["needs_review"]
    assert c.factory.vector.get(first.extra_node_ids[0]).is_latest

    hits = await c.search(query="Lives in Beijing now", user_id="u", min_score=0)
    (flagged,) = [i for i in hits.memories.normal if i.memory_id == provisional.node_id]
    assert flagged.needs_review

    phase["review"] = True
    digest = await c.digest()
    assert digest.reviewed == 1 and c.pending_reviews() == []
    (current,) = [n for n in head(c, layer=Layer.L1_FACT)]
    assert current.content == "Lives in Beijing now"
    assert set(current.supersedes) == {first.extra_node_ids[0], provisional.node_id}
    assert not c.factory.vector.get(first.extra_node_ids[0]).is_latest
    assert "needs_review" not in (current.custom or {})
    await c.aclose()


async def test_review_confirming_supplement_only_clears_flag(tmp_storage, fake_embed):
    weak = JevAnswer(type="choice", choice="ADD_CONFLICT", confidence=0.3,
                     probabilities={"ADD_CONFLICT": 0.3})

    def script(state, questions):
        heads = state.get("current", [])
        if "p0:target" in questions:
            return {"p0:target": heads[0]["id"] if heads else "NONE"}
        if state.get("review"):
            return {}
        return {"p0:action": weak} if heads else {}

    jev = FakeJev(script)
    c = make(tmp_storage, fake_embed, None, jev, reconcile_review_confidence=0.5)
    await c.add(content="Likes tea", user_id="u")
    second = await c.add(content="Likes coffee", user_id="u")
    assert second.decisions[0].review_reason == "low-confidence ADD_CONFLICT"
    # On review the relation question is re-asked without the confidence route.
    jev.responses = lambda state, questions: (
        {"p0:target": "NONE"} if "p0:target" in questions else {}
    )
    assert (await c.digest()).reviewed == 1
    node = c.factory.vector.get(second.extra_node_ids[0])
    assert node.is_latest and "needs_review" not in (node.custom or {})
    assert len(head(c, layer=Layer.L1_FACT)) == 2
    await c.aclose()


@pytest.mark.parametrize(
    "answers,status,action",
    [
        ({"p0:action": "RAW_ONLY"}, "raw_only", "RAW_ONLY"),
        ({"p0:action": "INVALID"}, "raw_only", "INVALID"),
        ({}, "committed", "ADD"),
    ],
)
async def test_write_status_distinguishes_outcomes(
    tmp_storage, fake_embed, answers, status, action
):
    c = make(tmp_storage, fake_embed, {"facts": [{"content": "hello"}]}, FakeJev(answers))
    result = await c.add(content="hello", user_id="u")
    assert result.status == status and result.decisions[0].action == action
    assert result.is_ephemeral == (action == "RAW_ONLY")
    await c.aclose()


async def test_known_claim_reports_known_status(tmp_storage, fake_embed):
    c = make(tmp_storage, fake_embed, {"facts": [{"content": "Known"}]})
    first = await c.add(content="Known", user_id="u")
    c.factory.jev.responses = {"p0:action": "SKIP", "p0:target": first.extra_node_ids[0]}
    second = await c.add(content="Known", user_id="u")
    assert second.status == "known" and second.decisions[0].action == "SKIP"
    assert second.decisions[0].targets == first.extra_node_ids
    await c.aclose()


async def test_as_of_hides_expired_temporal_exception(tmp_storage, fake_embed):
    c = make(
        tmp_storage,
        fake_embed,
        {"facts": [{"content": "Vegetarian this week", "valid_from": "2026-10-12",
                    "valid_until": "2026-10-18", "temporal_scope": "this week"}]},
    )
    result = await c.add(content="I'm vegetarian this week", user_id="u")
    node = c.factory.vector.get(result.extra_node_ids[0])
    assert node.valid_until == 1792367999 and node.custom["temporal_scope"] == "this week"
    inside = await c.search(query="Vegetarian this week", user_id="u", min_score=0,
                            as_of=1792200000)
    after = await c.search(query="Vegetarian this week", user_id="u", min_score=0,
                           as_of=1792400000)
    unfiltered = await c.search(query="Vegetarian this week", user_id="u", min_score=0)
    assert [i.content for i in inside.memories.normal] == ["Vegetarian this week"]
    assert after.memories.normal == []
    assert len(unfiltered.memories.normal) == 1
    text = format_memories(inside.memories.to_dict())
    assert "(valid: 2026-10-12 → 2026-10-18)" in text
    await c.aclose()


async def test_malformed_validity_is_dropped_not_fatal(tmp_storage, fake_embed):
    c = make(tmp_storage, fake_embed,
             {"facts": [{"content": "Busy", "valid_until": "next week"}]})
    result = await c.add(content="busy", user_id="u")
    assert result.status == "committed"
    assert c.factory.vector.get(result.extra_node_ids[0]).valid_until is None
    await c.aclose()


async def test_conflicts_are_attached_and_rendered(tmp_storage, fake_embed):
    def script(state, questions):
        heads = state.get("current", [])
        if "p0:target" in questions:
            return {"p0:target": heads[0]["id"]}
        return {"p0:action": "ADD_CONFLICT"} if heads else {}

    c = make(tmp_storage, fake_embed, None, FakeJev(script))
    await c.add(content="Works at Acme", user_id="u")
    await c.add(content="Works at Globex", user_id="u")
    hits = await c.search(query="Works at Globex", user_id="u", min_score=0)
    globex = next(i for i in hits.memories.normal if i.content == "Works at Globex")
    assert [x.content for x in globex.conflicts] == ["Works at Acme"]
    line = format_search_hit_line(globex.to_dict() | {"memory": globex.content})
    assert "conflicts, both unretracted" in line and "Works at Acme" in line
    bare = await c.search(query="Works at Globex", user_id="u", min_score=0,
                          include_relations=False)
    assert all(i.conflicts is None for i in bare.memories.normal)
    await c.aclose()


async def test_raw_route_returns_scaled_fallback_evidence(tmp_storage, fake_embed):
    c = make(tmp_storage, fake_embed, {"facts": [{"content": "Bought a bike"}]})
    added = await c.add(content="I bought a used road bike for 320 dollars", user_id="u")
    off = await c.search(query="320 dollars bike", user_id="u", min_score=0)
    on = await c.search(query="320 dollars bike", user_id="u", min_score=0, raw_limit=3)
    assert off.memories.raw == []
    (raw,) = on.memories.raw
    assert raw.memory_id == added.memory_id and raw.layer == "L0_RAW"
    assert 0 < raw.score <= 0.5
    assert raw.memory_id in {i.memory_id for i in on.memories.flatten()}
    await c.aclose()


def test_api_maps_decision_errors(tmp_storage, fake_embed):
    class Failing(FakeJev):
        def __init__(self, exc):
            super().__init__()
            self.exc = exc

        async def evaluate(self, **kw):
            raise self.exc

    for exc, status in (
        (StaleDecisionError("moved"), 409),
        (DecisionContextError("rejected"), 422),
    ):
        settings = Settings(storage_dir=f"{tmp_storage}/{status}")
        client = MemoryClient(
            settings=settings, embed=fake_embed, llm=FakeLLMClient(), jev=Failing(exc)
        )
        api = TestClient(create_app(client=client, settings=settings))
        response = api.post("/v1/memories/", json={"content": "x", "user_id": "u"})
        assert response.status_code == status
        assert response.json()["error_type"] == type(exc).__name__


def test_api_add_returns_status_and_honours_request_id(tmp_storage, fake_embed):
    settings = Settings(storage_dir=tmp_storage)
    client = MemoryClient(settings=settings, embed=fake_embed, llm=FakeLLMClient())
    api = TestClient(create_app(client=client, settings=settings))
    body = {"content": "likes tea", "user_id": "u", "request_id": "r1"}
    first = api.post("/v1/memories/", json=body).json()
    again = api.post("/v1/memories/", json=body).json()
    assert first["status"] == "committed" and first["decisions"][0]["action"] == "ADD"
    assert again["extra_node_ids"] == first["extra_node_ids"]
    conflict = api.post("/v1/memories/", json=body | {"content": "likes coffee"})
    assert conflict.status_code == 409


async def test_request_outcomes_live_in_own_table_and_expire(tmp_storage, fake_embed):
    c = make(tmp_storage, fake_embed, {"facts": [{"content": "x"}]})
    await c.add(content="x", user_id="u", request_id="keep")
    cache = c.factory.cache
    key = 'request:[["default", "u", ""], "keep"]'
    assert cache.get_request_outcome(key) and cache.get_profile(key) is None
    assert cache.purge_request_outcomes(older_than=0) == 0
    c.settings.request_outcome_ttl_seconds = 1
    cache.conn.execute("UPDATE request_outcomes SET ts = 0")
    await c.digest()
    assert cache.get_request_outcome(key) is None
    await c.aclose()
