import json
from dataclasses import replace
import pytest
from conftest import FakeLLMClient, FakeJev
from dual_mem.registry import ComponentFactory
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode
from dual_mem.system2.system2_agent import System2Agent
from dual_mem.agent.reconciler import DecisionContextError


def factory(tmp_storage, embed, response=None, jev=None, cap=2):
    def default(system, user):
        facts = json.loads(user)["facts"]
        ids = [f["id"] for f in facts]
        return {
            "memories": [
                {
                    "content": "derived " + ",".join(ids),
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


async def test_all_facts_paged_with_exact_coverage_and_real_evidence(
    tmp_storage, fake_embed
):
    f = factory(tmp_storage, fake_embed)
    original = seed(f)
    stats = await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert stats["created_schemas"] == 3
    assert len(f.llm.calls) == 3 and len(f.jev.calls) == 9
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
        {"memories": [], "processed_ids": []},
        {
            "memories": [{"content": "invented", "evidence_ids": ["foreign"]}],
            "processed_ids": ["0"],
        },
        {"memories": [{"content": "", "evidence_ids": ["0"]}], "processed_ids": ["0"]},
        {"memories": "invalid", "processed_ids": ["0"]},
    ],
)
async def test_invalid_generation_leaves_uncovered_sources_fresh(
    tmp_storage, fake_embed, response
):
    f = factory(tmp_storage, fake_embed, response)
    seed(f, 1)
    with pytest.raises(DecisionContextError):
        await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert f.vector.get("0").s2_evidence_count == 0
    assert f.vector.get_many({"layer": Layer.L3_DERIVED.value}) == []
    f.close()


async def test_rejected_inference_not_acknowledged(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed, jev=FakeJev({"coverage": "INVALID"}))
    seed(f, 1)
    with pytest.raises(DecisionContextError, match="coverage"):
        await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert f.vector.get("0").s2_evidence_count == 0
    f.close()


async def test_no_inference_still_requires_jev_coverage_verdict(
    tmp_storage, fake_embed
):
    f = factory(tmp_storage, fake_embed, {"memories": [], "processed_ids": ["0"]})
    seed(f, 1)
    await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert len(f.jev.calls) == 1 and f.vector.get("0").s2_evidence_count == 1
    assert f.vector.get_many({"layer": Layer.L3_DERIVED.value}) == []
    f.close()


async def test_oversized_fact_retained_pending(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed)
    f.settings.llm_context_window = 1200
    f.settings.llm_completion_reserve = 100
    n = seed(f, 1)[0]
    n.content = "x" * 4000
    # seed initial oversized source directly in a new scope
    n.node_id = "long"
    f.vector.upsert([n])
    with pytest.raises(ValueError, match="context"):
        await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert f.vector.get("long").s2_evidence_count == 0
    f.close()


async def test_changed_evidence_cannot_commit_inference(tmp_storage, fake_embed):
    f = factory(tmp_storage, fake_embed)
    seed(f, 1)

    def response(system, user):
        f.vector.update_payload("0", {"is_latest": False})
        return {
            "memories": [
                {"content": "derived", "kind": "schema", "evidence_ids": ["0"]}
            ],
            "processed_ids": ["0"],
        }

    f.llm.responses = {"json": response}
    with pytest.raises(DecisionContextError, match="evidence changed"):
        await System2Agent(factory=f).run(app_id="app", user_id="u")
    assert f.vector.get("0").s2_evidence_count == 0
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
