import time

import pytest

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.reader import Reader
from dual_mem.types import Layer, MemoryNode


@pytest.fixture
def factory(tmp_storage, fake_embed):
    f = ComponentFactory(settings=Settings(mode="system1", storage_dir=tmp_storage))
    f._embed = fake_embed
    return f


def _seed(factory, content, layer, *, gmt_created=None, **kw):
    """Synchronous seed helper using fake_embed.embed_sync (no event loop)."""
    node = MemoryNode(
        content=content,
        layer=layer,
        app_id="app",
        user_id="u",
        gmt_created=gmt_created or int(time.time()),
        **kw,
    )
    node.embedding = factory.embed.embed_sync(content)
    factory.vector.upsert([node])
    return node


async def test_three_key_structure_and_profile_full(factory):
    p0 = _seed(factory, "用户叫张三", Layer.L0_BASIC_INFO)
    p4 = _seed(factory, "用户是工程师", Layer.L4_IDENTITY)
    f1 = _seed(factory, "用户喜欢喝咖啡", Layer.L2_FACT)
    f2 = _seed(factory, "用户住在北京", Layer.L2_FACT)

    reader = Reader(factory=factory)
    res = await reader.search(query="用户喜欢喝咖啡", app_ids=["app"], user_id="u", limit=5)

    assert {"profile", "proactive", "normal"} == {"profile", "proactive", "normal"}
    assert res.proactive == []
    profile_ids = {m.memory_id for m in res.profile}
    assert profile_ids == {p0.node_id, p4.node_id}
    normal_ids = {m.memory_id for m in res.normal}
    assert f1.node_id in normal_ids
    assert f2.node_id in normal_ids


async def test_normal_respects_limit(factory):
    for i in range(5):
        _seed(factory, f"事实条目 {i}", Layer.L2_FACT)

    reader = Reader(factory=factory)
    res = await reader.search(
        query="事实条目 0", app_ids=["app"], user_id="u", limit=2, min_score=0.0
    )
    assert len(res.normal) == 2


async def test_evolution_chain_returned(factory):
    # A(latest) -> B -> C(oldest)
    c = _seed(
        factory, "旧版偏好：喜欢茶", Layer.L2_FACT, gmt_created=100, is_latest=False
    )
    b = _seed(
        factory,
        "中间版偏好：喜欢茶和咖啡",
        Layer.L2_FACT,
        gmt_created=200,
        is_latest=False,
    )
    a = _seed(
        factory,
        "最新偏好：喜欢咖啡",
        Layer.L2_FACT,
        gmt_created=300,
        is_latest=True,
    )
    a.supersedes = [b.node_id]
    b.supersedes = [c.node_id]
    b.superseded_by = [a.node_id]
    c.superseded_by = [b.node_id]
    factory.vector.upsert([a, b, c])

    reader = Reader(factory=factory)
    res = await reader.search(
        query="中间版偏好：喜欢茶和咖啡",
        app_ids=["app"],
        user_id="u",
        limit=5,
        min_score=0.0,
    )
    evolved = [m for m in res.normal if m.evolution_chain]
    assert len(evolved) == 1
    assert evolved[0].memory_id == a.node_id
    chain_ids = [item.node_id for item in evolved[0].evolution_chain]
    assert chain_ids == [a.node_id, b.node_id, c.node_id]


async def test_created_after_filter(factory):
    old = _seed(factory, "去年的旧事实", Layer.L2_FACT, gmt_created=1000)
    new = _seed(factory, "今天的新事实", Layer.L2_FACT, gmt_created=9_000_000_000)

    reader = Reader(factory=factory)
    res = await reader.search(
        query="今天的新事实",
        app_ids=["app"],
        user_id="u",
        limit=5,
        min_score=0.0,
        created_after=5_000_000_000,
    )
    ids = {m.memory_id for m in res.normal}
    assert new.node_id in ids
    assert old.node_id not in ids
