import pytest

from dual_mem.isolation import build_filter
from dual_mem.storage.vector_store import ChromaVectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus


def _node(fake_embed, content, layer=Layer.L2_FACT, app_id="app", user_id="u"):
    node = MemoryNode(content=content, layer=layer, app_id=app_id, user_id=user_id)
    node.embedding = fake_embed.embed(content)
    return node


@pytest.fixture
def store(tmp_storage):
    return ChromaVectorStore(tmp_storage)


def test_upsert_and_query_hit(store, fake_embed):
    n1 = _node(fake_embed, "用户喜欢喝咖啡")
    n2 = _node(fake_embed, "用户住在北京")
    store.upsert([n1, n2])

    where = build_filter(app_ids=["app"], user_id="u", statuses=[MemoryStatus.ACTIVE])
    results = store.query(embedding=fake_embed.embed("用户喜欢喝咖啡"), where=where, top_k=5)
    assert results[0].node_id == n1.node_id
    assert results[0].score > 0.99


def test_get_and_delete(store, fake_embed):
    n1 = _node(fake_embed, "fact one")
    store.upsert([n1])
    got = store.get(n1.node_id)
    assert got is not None
    assert got.content == "fact one"

    store.delete([n1.node_id])
    assert store.get(n1.node_id) is None


def test_update_status_to_shadow(store, fake_embed):
    n1 = _node(fake_embed, "fact shadow")
    store.upsert([n1])
    store.update_status(n1.node_id, MemoryStatus.SHADOW)
    got = store.get(n1.node_id)
    assert got.status is MemoryStatus.SHADOW


def test_get_many_filter(store, fake_embed):
    n1 = _node(fake_embed, "fact a", layer=Layer.L2_FACT)
    n2 = _node(fake_embed, "summary b", layer=Layer.L3_SUMMARY)
    store.upsert([n1, n2])

    where = build_filter(app_ids=["app"], user_id="u", layers=[Layer.L2_FACT])
    nodes = store.get_many(where)
    assert len(nodes) == 1
    assert nodes[0].node_id == n1.node_id


def test_update_payload_superseded_by(store, fake_embed):
    n1 = _node(fake_embed, "old fact")
    store.upsert([n1])
    store.update_payload(n1.node_id, {"superseded_by": "node-x", "is_latest": False})
    got = store.get(n1.node_id)
    assert got.superseded_by == ["node-x"]
    assert got.is_latest is False
