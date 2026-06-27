import pytest

from dual_mem.storage.graph_store import GraphNode, KuzuGraphStore


@pytest.fixture
def gstore(tmp_storage):
    return KuzuGraphStore(tmp_storage)


def _gnode(fake_embed, nid, content, tags=None):
    return GraphNode(
        node_id=nid,
        layer="L6_SCHEMA",
        content=content,
        app_id="app",
        user_id="u",
        embedding=fake_embed.embed_sync(content),
        tags=tags or [],
    )


def test_add_and_query_by_embedding(gstore, fake_embed):
    gstore.add_node(_gnode(fake_embed, "s1", "用户偏好简洁回答"))
    gstore.add_node(_gnode(fake_embed, "s2", "用户喜欢详细的长篇解释"))

    res = gstore.query_by_embedding(
        layer="L6_SCHEMA",
        user_id="u",
        app_ids=["app"],
        embedding=fake_embed.embed_sync("用户偏好简洁回答"),
        top_k=5,
    )
    assert res[0].node_id == "s1"
    assert res[0].score > 0.99


def test_add_evidence_and_evidence_of(gstore, fake_embed):
    gstore.add_node(_gnode(fake_embed, "s1", "schema node"))
    gstore.add_evidence(schema_id="s1", fact_id="f1")
    gstore.add_evidence(schema_id="s1", fact_id="f2")
    assert set(gstore.evidence_of("s1")) == {"f1", "f2"}


def test_evidence_counts_batched(gstore, fake_embed):
    """evidence_counts returns per-schema DERIVED_FROM counts in one query."""
    gstore.add_node(_gnode(fake_embed, "s1", "schema one"))
    gstore.add_node(_gnode(fake_embed, "s2", "schema two"))
    gstore.add_node(_gnode(fake_embed, "s3", "schema three (no evidence)"))
    gstore.add_evidence(schema_id="s1", fact_id="f1")
    gstore.add_evidence(schema_id="s1", fact_id="f2")
    gstore.add_evidence(schema_id="s2", fact_id="f3")

    counts = gstore.evidence_counts(["s1", "s2", "s3"])
    assert counts == {"s1": 2, "s2": 1}
    assert gstore.evidence_counts([]) == {}


def test_tag_bridge(gstore, fake_embed):
    gstore.add_node(_gnode(fake_embed, "s1", "node a", tags=["coffee"]))
    gstore.add_node(_gnode(fake_embed, "s2", "node b", tags=["coffee"]))
    gstore.add_node(_gnode(fake_embed, "s3", "node c", tags=["tea"]))

    ids = gstore.neighbors_by_tag(tag="coffee", user_id="u", app_ids=["app"])
    assert set(ids) == {"s1", "s2"}


def test_add_edge(gstore, fake_embed):
    gstore.add_node(_gnode(fake_embed, "s1", "node a"))
    gstore.add_node(_gnode(fake_embed, "s2", "node b"))
    gstore.add_edge(from_id="s1", to_id="s2", rel="RELATED_TO")


def test_list_by_layer_and_custom_roundtrip(gstore, fake_embed):
    basic = _gnode(fake_embed, "s1", "basic schema")
    core = _gnode(fake_embed, "s2", "core schema")
    core.custom = {"sub_type": "core"}
    gstore.add_node(basic)
    gstore.add_node(core)

    nodes = gstore.list_by_layer(layer="L6_SCHEMA", user_id="u", app_ids=["app"])
    assert {n.node_id for n in nodes} == {"s1", "s2"}
    by_id = {n.node_id: n for n in nodes}
    assert by_id["s1"].custom is None
    assert by_id["s2"].custom == {"sub_type": "core"}
