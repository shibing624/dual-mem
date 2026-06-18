import dataclasses

from dual_mem.types import (
    LAYER_TO_CATEGORY,
    Category,
    Layer,
    MemoryNode,
    MemoryStatus,
    ReconcileOp,
)


def test_layer_has_eight_members():
    assert len(Layer) == 8
    expected = {
        "L0_BASIC_INFO",
        "L1_RAW",
        "L2_FACT",
        "L3_SUMMARY",
        "L4_IDENTITY",
        "L5_KNOWLEDGE",
        "L6_SCHEMA",
        "L7_INTENTION",
    }
    assert {m.name for m in Layer} == expected
    for m in Layer:
        assert m.value == m.name


def test_reconcile_op_only_three():
    assert {m.name for m in ReconcileOp} == {"ADD", "SUPERSEDE", "DELETE"}


def test_status_members():
    assert {m.name for m in MemoryStatus} == {
        "ACTIVE",
        "SHADOW",
        "SUPERSEDED",
        "DELETED",
    }


def test_layer_to_category_map():
    assert LAYER_TO_CATEGORY[Layer.L0_BASIC_INFO] == Category.profile
    assert LAYER_TO_CATEGORY[Layer.L1_RAW] == Category.raw
    assert LAYER_TO_CATEGORY[Layer.L2_FACT] == Category.fact
    assert LAYER_TO_CATEGORY[Layer.L3_SUMMARY] == Category.summary
    assert LAYER_TO_CATEGORY[Layer.L4_IDENTITY] == Category.profile
    assert LAYER_TO_CATEGORY[Layer.L5_KNOWLEDGE] == Category.knowledge
    assert LAYER_TO_CATEGORY[Layer.L6_SCHEMA] == Category.schema
    assert LAYER_TO_CATEGORY[Layer.L7_INTENTION] == Category.intention


def test_node_defaults_and_category():
    n = MemoryNode(content="hi", layer=Layer.L2_FACT, app_id="app", user_id="u")
    assert n.status == MemoryStatus.ACTIVE
    assert n.is_latest is True
    assert n.node_id
    assert n.gmt_created > 0
    assert n.category == Category.fact
    assert n.score == 0.0


def test_score_not_in_eq():
    a = MemoryNode(content="x", layer=Layer.L1_RAW, app_id="a", user_id="u", node_id="fixed")
    b = MemoryNode(content="x", layer=Layer.L1_RAW, app_id="a", user_id="u", node_id="fixed",
                   gmt_created=a.gmt_created)
    b.score = 9.9
    assert a == b


def test_to_metadata_scalar_only():
    n = MemoryNode(
        content="c",
        layer=Layer.L2_FACT,
        app_id="app",
        user_id="u",
        tags=["t1", "t2"],
        supersedes=["s1"],
    )
    meta = n.to_metadata()
    for v in meta.values():
        assert isinstance(v, (str, int, float, bool))
    assert meta["tags"] == "t1\x1ft2"
    assert meta["supersedes"] == "s1"
    assert meta["memory_at"] == -1
    assert meta["gmt_modified"] == -1
    assert meta["speculate"] == ""


def test_roundtrip_from_storage():
    n = MemoryNode(
        content="content text",
        layer=Layer.L7_INTENTION,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        tags=["a", "b"],
        supersedes=["x"],
        superseded_by=["y"],
        speculate="maybe",
        memory_at=123,
        gmt_modified=456,
        custom={"basic_info_kv": {"name": "张三"}},
    )
    meta = n.to_metadata()
    assert isinstance(meta["custom"], str)
    restored = MemoryNode.from_storage(n.content, meta, embedding=[0.1, 0.2])
    assert restored.custom == {"basic_info_kv": {"name": "张三"}}
    assert restored.content == n.content
    assert restored.layer == n.layer
    assert restored.tags == ["a", "b"]
    assert restored.supersedes == ["x"]
    assert restored.superseded_by == ["y"]
    assert restored.speculate == "maybe"
    assert restored.memory_at == 123
    assert restored.gmt_modified == 456
    assert restored.embedding == [0.1, 0.2]
    assert restored.status == n.status


def test_roundtrip_none_fields():
    n = MemoryNode(content="c", layer=Layer.L1_RAW, app_id="app", user_id="u")
    restored = MemoryNode.from_storage(n.content, n.to_metadata())
    assert restored.memory_at is None
    assert restored.gmt_modified is None
    assert restored.speculate is None
    assert restored.tags == []
    assert restored.custom is None
