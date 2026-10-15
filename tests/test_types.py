import dataclasses

from dual_mem.types import (
    LAYER_TO_CATEGORY,
    Category,
    Layer,
    MemoryNode,
    MemoryStatus,
    ReconcileOp,
)


def test_four_canonical_layers_with_legacy_read_aliases():
    assert {m.name for m in Layer} == {"L0_RAW", "L1_FACT", "L2_PROFILE", "L3_DERIVED"}
    assert Layer("L2_FACT") == Layer.L1_FACT
    assert Layer.L4_IDENTITY == Layer.L2_PROFILE
    assert Layer("L6_SCHEMA") == Layer.L3_DERIVED


def test_reconcile_actions_include_skip():
    assert {m.name for m in ReconcileOp} == {"ADD", "DELETE", "SKIP", "SUPERSEDE"}


def test_status_members():
    assert {m.name for m in MemoryStatus} == {
        "ACTIVE",
        "SHADOW",
        "SUPERSEDED",
        "DELETED",
    }


def test_category_uses_kind_with_four_layer_routing():
    assert LAYER_TO_CATEGORY[Layer.L0_RAW] == Category.raw
    assert LAYER_TO_CATEGORY[Layer.L1_FACT] == Category.fact
    assert LAYER_TO_CATEGORY[Layer.L2_PROFILE] == Category.profile
    assert (
        MemoryNode(
            content="plan",
            layer=Layer.L1_FACT,
            kind="intention",
            app_id="a",
            user_id="u",
        ).category
        == Category.intention
    )
    assert (
        MemoryNode(
            content="summary",
            layer=Layer.L3_DERIVED,
            kind="summary",
            app_id="a",
            user_id="u",
        ).category
        == Category.summary
    )


def test_node_defaults_and_category():
    n = MemoryNode(content="hi", layer=Layer.L2_FACT, app_id="app", user_id="u")
    assert n.status == MemoryStatus.ACTIVE
    assert n.is_latest is True
    assert n.node_id
    assert n.gmt_created > 0
    assert n.category == Category.fact
    assert n.score == 0.0


def test_score_not_in_eq():
    a = MemoryNode(
        content="x", layer=Layer.L1_RAW, app_id="a", user_id="u", node_id="fixed"
    )
    b = MemoryNode(
        content="x",
        layer=Layer.L1_RAW,
        app_id="a",
        user_id="u",
        node_id="fixed",
        gmt_created=a.gmt_created,
    )
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
