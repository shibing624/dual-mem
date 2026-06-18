from conftest import FakeLLMClient

from dual_mem import MemoryClient
from dual_mem.types import Layer, MemoryNode

_SCHEMA = "当处理任务时，用户追求确定性与掌控——反映低不确定性容忍。"
_INTENTION = "用户正在准备一场数据库技术分享。"

_OPS = [
    {"op": "create_schema", "content": _SCHEMA, "tags": ["工作"], "evidence": ["a1", "a2", "a3"]},
    {"op": "create_intention", "content": _INTENTION, "tags": ["工作"], "evidence": ["a1"]},
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


async def test_ultra_digest_then_recall_schema_and_intention(tmp_storage, fake_embed):
    client = MemoryClient(
        storage_dir=tmp_storage,
        mode="ultra",
        embed=fake_embed,
        llm=FakeLLMClient(responses={"json": _OPS}),
    )
    _seed_fresh_facts(client)
    client.factory.cache.enqueue_s2_task("u", "app")

    digest = await client.digest()
    assert digest["processed"] == 1

    # profile 路召回 L6 schema
    res = await client.search(query=_SCHEMA, app_ids=["app"], user_id="u")
    profile_contents = [m["content"] for m in res["memories"]["profile"]]
    assert _SCHEMA in profile_contents
    assert any(m["category"] == "schema" for m in res["memories"]["profile"])

    # proactive 路（intention_limit>0）召回 L7 intention
    res2 = await client.search(
        query=_INTENTION, app_ids=["app"], user_id="u", intention_limit=3
    )
    proactive_contents = [m["content"] for m in res2["memories"]["proactive"]]
    assert _INTENTION in proactive_contents

    # 默认 intention_limit=0 → proactive 恒空
    res3 = await client.search(query=_INTENTION, app_ids=["app"], user_id="u")
    assert res3["memories"]["proactive"] == []
