import json

from conftest import FakeLLMClient

from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode

_SCHEMA = "当处理任务时，用户追求确定性与掌控——反映低不确定性容忍。"
_INTENTION = "用户正在准备一场数据库技术分享。"


def _tc(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


# Scripted ReAct turns: create the schema, create the intention, then stop.
_TOOL_TURNS = [
    {"content": "", "tool_calls": [
        _tc("c1", "create_schema",
            {"content": _SCHEMA, "tags": ["工作"], "evidence": ["a1", "a2", "a3"]}),
        _tc("c2", "create_intention",
            {"content": _INTENTION, "tags": ["工作"], "evidence": ["a1"]}),
    ]},
    {"content": "", "tool_calls": []},
]


def _vec(comps: dict[int, float], dim: int = 64) -> list[float]:
    v = [0.0] * dim
    for i, val in comps.items():
        v[i] = val
    return v


def _seed_fresh_facts(client):
    nodes = []
    for nid, content, comp in [
        ("a1", "用户做项目前先列详细计划", {0: 1.0, 4: 0.5}),
        ("a2", "用户写代码前先画架构图", {0: 1.0, 5: 0.5}),
        ("a3", "用户旅行前把行程排满", {0: 1.0, 6: 0.5}),
    ]:
        node = MemoryNode(
            content=content, layer=Layer.L2_FACT, app_id="app", user_id="u", node_id=nid
        )
        node.embedding = _vec(comp)
        nodes.append(node)
    client.factory.vector.upsert(nodes)


async def test_dual_digest_then_recall_schema_and_intention(tmp_storage, fake_embed):
    # Isolate the L6/L7 plumbing: force the ReAct loop (scripted tool turns) and keep
    # derived-layer recall on regardless of the (FACTUAL) query intent.
    settings = Settings(
        mode="dual",
        storage_dir=tmp_storage,
        system2_single_shot_max_clusters=0,
    )
    client = MemoryClient(
        settings=settings,
        embed=fake_embed,
        llm=FakeLLMClient(responses={"tools": _TOOL_TURNS}),
    )
    _seed_fresh_facts(client)
    client.factory.cache.enqueue_s2_task("u", "app")

    digest = await client.digest()
    assert digest.processed >= 1

    # profile 路召回 L6 schema
    res = await client.search(query=_SCHEMA, app_ids=["app"], user_id="u")
    profile_contents = [m.content for m in res.memories.profile]
    assert _SCHEMA in profile_contents
    assert any(m.category == "schema" for m in res.memories.profile)

    # proactive 路（intention_limit>0）召回 L7 intention
    res2 = await client.search(
        query=_INTENTION, app_ids=["app"], user_id="u", intention_limit=3
    )
    proactive_contents = [m.content for m in res2.memories.proactive]
    assert _INTENTION in proactive_contents

    # 默认 intention_limit=0 → proactive 恒空
    res3 = await client.search(query=_INTENTION, app_ids=["app"], user_id="u")
    assert res3.memories.proactive == []


async def test_explicitly_excluding_derived_schema(
    tmp_storage,
    fake_embed,
    monkeypatch,
):
    """include_derived=False excludes L6 schema without guessing query intent."""
    settings = Settings(
        mode="dual",
        storage_dir=tmp_storage,
        system2_single_shot_max_clusters=0,
    )
    client = MemoryClient(
        settings=settings,
        embed=fake_embed,
        llm=FakeLLMClient(responses={"tools": _TOOL_TURNS}),
    )
    _seed_fresh_facts(client)
    client.factory.cache.enqueue_s2_task("u", "app")
    await client.digest()

    graph_queries = 0
    original_query = client.factory.graph.query_by_embedding

    def _track_query(**kwargs):
        nonlocal graph_queries
        graph_queries += 1
        return original_query(**kwargs)

    monkeypatch.setattr(client.factory.graph, "query_by_embedding", _track_query)

    res = await client.search(
        query=_SCHEMA,
        app_ids=["app"],
        user_id="u",
        intention_limit=3,
        include_derived=False,
    )
    assert all(m.category != "schema" for m in res.memories.profile)
    assert res.memories.proactive == []
    assert graph_queries == 0
