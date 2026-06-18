import pytest

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.reader import Reader
from dual_mem.types import Layer, MemoryNode


@pytest.fixture
def factory(tmp_storage, fake_embed):
    f = ComponentFactory(settings=Settings(mode="lite", storage_dir=tmp_storage))
    f._embed = fake_embed
    return f


def _seed(factory, content):
    node = MemoryNode(content=content, layer=Layer.L2_FACT, app_id="app", user_id="u")
    node.embedding = factory.embed.embed(content)
    factory.vector.upsert([node])
    return node


def test_normal_hit_profile_proactive_empty(factory):
    n1 = _seed(factory, "用户喜欢喝咖啡")
    _seed(factory, "用户住在北京")

    reader = Reader(factory=factory)
    result = reader.search(
        query="用户喜欢喝咖啡", app_ids=["app"], user_id="u", limit=5
    )

    assert result["profile"] == []
    assert result["proactive"] == []
    assert result["normal"][0]["memory_id"] == n1.node_id
    assert result["normal"][0]["category"] == "fact"
    assert result["normal"][0]["score"] >= 0.99
