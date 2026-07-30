"""Tests for the System2 / read-path optimizations:

- s2 single-shot path (clusters<=1 → one chat_json, no ReAct tool loop)
- reconcile no/weak-candidate fast path (skip LLM, SUPPLEMENT)
- reconcile_policy=conservative prompt addendum
- lossless shadow (uncovered fast-write originals stay ACTIVE)
- include_evolution-by-intent heuristic
"""
import json

import pytest

from dual_mem.agent.reconciler import Reconciler
from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.formatter import include_evolution_for_query
from dual_mem.retrieval.intent import wants_evolution_history
from dual_mem.system2.reconciler_worker import ReconcilerWorker
from dual_mem.system2.system2_agent import System2Agent
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from tests.conftest import FakeLLMClient


def _vec(comps: dict[int, float], dim: int = 16) -> list[float]:
    v = [0.0] * dim
    for i, val in comps.items():
        v[i] = val
    return v


def _seed_one_cluster(factory):
    nodes = []
    for nid, content, comp in [
        ("a1", "用户做饭严格按菜谱", {0: 1.0, 4: 0.5}),
        ("a2", "用户烘焙精确称量", {0: 1.0, 5: 0.5}),
        ("a3", "用户煮咖啡按固定比例", {0: 1.0, 6: 0.5}),
    ]:
        node = MemoryNode(
            content=content, layer=Layer.L2_FACT, app_id="app", user_id="u", node_id=nid
        )
        node.embedding = _vec(comp)
        nodes.append(node)
    factory.vector.upsert(nodes)


# --------------------------------------------------------------------------- single-shot


async def test_single_shot_one_cluster_no_react(tmp_storage, fake_embed):
    """One cluster → single chat_json emitting ops; the ReAct tool loop is never entered."""
    settings = Settings(
        mode="dual", storage_dir=tmp_storage, system2_single_shot_max_clusters=1
    )
    ops = {
        "ops": [
            {
                "op": "create_schema",
                "content": "当做饭时，用户严格按菜谱——反映对结构的需要。",
                "tags": ["烹饪"],
                "evidence": ["a1", "a2", "a3"],
            }
        ]
    }
    llm = FakeLLMClient(responses={"json": ops})
    factory = ComponentFactory(settings=settings, embed=fake_embed, llm=llm)
    _seed_one_cluster(factory)

    result = await System2Agent(factory=factory).run(app_id="app", user_id="u")
    assert result["created_schemas"] == 1
    assert result["evidence_added"] == 3
    # No ReAct tool round-trips: single-shot uses chat_json only.
    assert all(c["type"] != "chat_with_tools" for c in llm.calls)
    assert any(c["type"] == "chat_json" for c in llm.calls)


# --------------------------------------------------------------- reconcile weak/no candidate


async def test_reconcile_no_candidate_skips_llm(tmp_storage, fake_embed):
    """Empty candidate set → reconcile returns SUPPLEMENT ADDs without any LLM call."""
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    llm = FakeLLMClient()
    factory = ComponentFactory(settings=settings, embed=fake_embed, llm=llm)

    reconciler = Reconciler(
        llm=llm, embed=factory.embed, vector=factory.vector, weak_candidate_score=0.5
    )
    ops = await reconciler.reconcile(
        new_memories=["全新的事实A", "全新的事实B"],
        new_memories_meta=[
            {"content": "全新的事实A", "layer": "L2_FACT", "tags": []},
            {"content": "全新的事实B", "layer": "L2_FACT", "tags": []},
        ],
        app_id="app",
        user_id="u",
        agent_id="",
        current_time="",
    )
    assert len(ops) == 2
    assert all(op.op == "ADD" and op.update_type == "SUPPLEMENT" and not op.supersedes for op in ops)
    # LLM must not have been consulted on the fast path.
    assert all(c["type"] != "chat_json" or "记忆管理系统" not in c["system"] for c in llm.calls)


# ------------------------------------------------------------------ conservative policy prompt


async def test_conservative_policy_appends_addendum(tmp_storage, fake_embed):
    """policy=conservative injects the no-merge hard rules into the reconcile system prompt."""
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    llm = FakeLLMClient(responses={"reconcile": []})
    factory = ComponentFactory(settings=settings, embed=fake_embed, llm=llm)

    # Seed an existing node IDENTICAL to the new memory so it recalls with cosine 1.0 (>0.5),
    # bypassing the weak-candidate fast path and reaching the LLM.
    existing = MemoryNode(
        content="用户喜欢苹果", layer=Layer.L2_FACT, app_id="app", user_id="u", node_id="e1"
    )
    existing.embedding = factory.embed.embed_sync("用户喜欢苹果")
    factory.vector.upsert([existing])

    reconciler = Reconciler(
        llm=llm, embed=factory.embed, vector=factory.vector, policy="conservative"
    )
    await reconciler.reconcile(
        new_memories=["用户喜欢苹果"],
        new_memories_meta=[{"content": "用户喜欢苹果", "layer": "L2_FACT", "tags": []}],
        app_id="app",
        user_id="u",
        agent_id="",
        current_time="",
    )
    recon_calls = [c for c in llm.calls if c["type"] == "chat_json" and "记忆管理系统" in c["system"]]
    assert recon_calls, "expected a reconcile LLM call"
    assert "保守策略" in recon_calls[0]["system"]


# ----------------------------------------------------------------------- lossless shadow


async def test_uncovered_original_stays_active(tmp_storage, fake_embed):
    """A reconcile ADD that does NOT re-emit an original's content must NOT shadow it."""
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    # reconcile merges everything into one unrelated "MERGED" node (does not cover originals).
    llm = FakeLLMClient(
        responses={"reconcile": [{"op": "ADD", "content": "MERGED", "layer": "L2_FACT"}]}
    )
    factory = ComponentFactory(settings=settings, embed=fake_embed, llm=llm)

    # Two fast-write originals (the task batch) + a pre-existing node identical to one of
    # them so the candidate set is non-empty and strong (cosine 1.0) → LLM path, not fast path.
    for nid, content in [("fw0", "事实零"), ("fw1", "事实一"), ("ex", "事实零")]:
        n = MemoryNode(
            content=content, layer=Layer.L2_FACT, app_id="app", user_id="u",
            status=MemoryStatus.ACTIVE, is_latest=True, node_id=nid,
        )
        n.embedding = factory.embed.embed_sync(content)
        factory.vector.upsert([n])
    factory.cache.enqueue_reconcile_task(
        app_id="app", user_id="u", agent_id="", node_ids=["fw0", "fw1"]
    )

    await ReconcilerWorker(factory=factory).reconcile_pending(
        app_id="app", user_id="u", agent_id=""
    )

    # "MERGED" did not cover either original → both remain ACTIVE (no silent loss).
    assert factory.vector.get("fw0").status is MemoryStatus.ACTIVE
    assert factory.vector.get("fw1").status is MemoryStatus.ACTIVE


# ------------------------------------------------------------------ evolution-by-intent


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What laptop do I use now?", False),
        ("What laptop did I use previously?", True),
        ("我之前用的是什么笔记本？", True),
        ("我现在用什么笔记本？", False),
        ("用户原来住在哪里", True),
    ],
)
def test_include_evolution_for_query(query, expected):
    assert include_evolution_for_query(query) is expected
    assert wants_evolution_history(query) is expected
