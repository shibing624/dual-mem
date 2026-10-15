from dataclasses import replace
import pytest
from dual_mem.storage.vector_store import ChromaVectorStore
from dual_mem.storage.history_store import HistoryStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus


def node(nid, **kw):
    return MemoryNode(
        node_id=nid,
        content=nid,
        layer=Layer.L1_FACT,
        app_id="app",
        user_id="u",
        embedding=[1.0, 0.0],
        **kw,
    )


@pytest.fixture
def store(tmp_storage):
    return ChromaVectorStore(tmp_storage, history=HistoryStore(tmp_storage))


def apply(store, nodes, expected=None, archives=None, epoch=None):
    return store.apply_version_plan(
        nodes=nodes,
        expected=expected or {},
        archives=archives or [],
        scope=("app", "u", ""),
        expected_epoch=epoch,
    )


def test_revisions_scope_and_immutable_history(store):
    old = node("a")
    store.upsert([old])
    epoch = store.scope_epoch(("app", "u", ""))
    assert apply(store, [node("b", supersedes=["a"])], {"a": 0}, epoch=epoch)
    a, b = store.get("a"), store.get("b")
    assert a.content == "a" and a.status == MemoryStatus.ACTIVE and not a.is_latest
    assert b.supersedes == ["a"] and a.superseded_by == ["b"]
    assert not apply(store, [node("c", supersedes=["a"])], {"a": 0}, epoch=epoch)
    assert store.get("c") is None


def test_scope_epoch_prevents_parallel_empty_scope_adds(store):
    epoch = store.scope_epoch(("app", "u", ""))
    assert apply(store, [node("a")], epoch=epoch)
    assert not apply(store, [node("b")], epoch=epoch)
    assert store.get("b") is None


def test_stable_replay_rejects_changed_links_and_metadata(store):
    store.upsert([node("a"), node("x")])
    b = node("b", supersedes=["a"])
    assert apply(store, [b], {"a": 0, "x": 0})
    assert apply(store, [b], {"a": 0, "x": 0})
    with pytest.raises(ValueError, match="semantics"):
        apply(store, [replace(b, supersedes=["x"])], {"a": 0, "x": 0})
    assert store.get("x").is_latest


@pytest.mark.parametrize(
    "new",
    [
        node("b", embedding=None) if False else replace(node("b"), embedding=[1.0]),
        replace(node("b"), embedding=[float("nan"), 0.0]),
        replace(node("b"), user_id="foreign"),
        node("b", supersedes=["missing"]),
    ],
)
def test_invalid_plan_does_not_poison_journal_or_write_anything(store, new):
    store.upsert([node("a")])
    with pytest.raises(ValueError):
        apply(store, [new], {"a": 0})
    assert store.get("a").is_latest and store.get("b") is None
    assert (
        store._journal.execute("SELECT count(*) FROM plans WHERE done=0").fetchone()[0]
        == 0
    )


def test_link_failure_recovers_before_next_read_with_exact_final_audit(
    store, monkeypatch
):
    store.upsert([node("a")])
    original = store.collection.update
    failed = False

    def transient(**kw):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated crash after append")
        return original(**kw)

    monkeypatch.setattr(store.collection, "update", transient)
    with pytest.raises(OSError):
        apply(
            store, [node("b", supersedes=["a"]), node("c", supersedes=["b"])], {"a": 0}
        )
    assert not store.get("a").is_latest
    b = store.get("b")
    assert not b.is_latest and b.superseded_by == ["c"] and b.revision == 1
    events = store._history.list_for_node("b")
    assert len(events) == 1 and not events[0]["new"]["is_latest"]
    assert apply(
        store, [node("b", supersedes=["a"]), node("c", supersedes=["b"])], {"a": 0}
    )
    assert len(store._history.list_for_node("b")) == 1


def test_direct_import_and_metadata_mutation_invalidate_material_snapshot(store):
    store.upsert([node("a")])
    epoch = store.scope_epoch(("app", "u", ""))
    store.update_status("a", MemoryStatus.SHADOW)
    assert store.get("a").revision == 1
    assert not apply(store, [node("b")], epoch=epoch)
    assert not apply(store, [node("b", supersedes=["a"])], {"a": 1})
    with pytest.raises(ValueError, match="immutable"):
        store.upsert([replace(node("a"), content="changed")])


def test_four_layer_migration_repairs_keep_active_ancestors(store):
    a, b = node("a", superseded_by=["b"]), node("b", supersedes=["a"])
    store.upsert([a, b])
    for n in (a, b):
        meta = n.to_metadata()
        meta["layer"] = "L2_FACT"
        meta["layer_schema_version"] = 2
        store.collection.update(ids=[n.node_id], metadatas=[meta])
    store._migrate_layers()
    assert not store.get("a").is_latest and store.get("b").is_latest
    assert store.get("a").status == MemoryStatus.ACTIVE
    assert store.get("a").layer == Layer.L1_FACT


def test_double_consumption_and_forward_dependency_rejected_before_append(store):
    store.upsert([node("a")])
    with pytest.raises(ValueError, match="two operations"):
        apply(
            store, [node("b", supersedes=["a"]), node("c", supersedes=["a"])], {"a": 0}
        )
    with pytest.raises(ValueError, match="forward"):
        apply(store, [node("b", supersedes=["c"]), node("c")])
    assert store.get("b") is None


def test_retired_source_invalidates_derived_and_refreshes_remaining_evidence(store):
    store.upsert(
        [
            node("a"),
            node("x", s2_evidence_count=1),
            replace(
                node("d"), layer=Layer.L3_DERIVED, custom={"source_ids": ["a", "x"]}
            ),
        ]
    )
    assert apply(store, [node("b", supersedes=["a"])], {"a": 0})
    assert not store.get("d").is_latest and store.get("x").s2_evidence_count == 0
    assert store.get("d").content == "d"


def test_empty_collection_still_rejects_wrong_dimension_before_journaling(store):
    store.upsert([node("a")])
    store.delete(["a"])
    with pytest.raises(ValueError, match="dimension"):
        apply(store, [replace(node("b"), embedding=[1.0, 0.0, 0.0])])
    assert (
        store._journal.execute("SELECT count(*) FROM plans WHERE done=0").fetchone()[0]
        == 0
    )
    assert apply(store, [node("c")])


def test_stale_import_replay_cannot_resurrect_version_head(store):
    store.upsert([node("a")])
    stale = store.get("a")
    assert apply(store, [node("b", supersedes=["a"])], {"a": 0})
    epoch = store.scope_epoch(("app", "u", ""))
    store.upsert([stale])
    assert not store.get("a").is_latest
    assert store.get("a").superseded_by == ["b"]
    assert store.scope_epoch(("app", "u", "")) == epoch


def test_legacy_one_way_chain_is_repaired_before_migration_certification(store):
    a, b = node("a"), node("b", supersedes=["a"])
    store.upsert([a, b])
    for n in (a, b):
        meta = n.to_metadata()
        meta["layer_schema_version"] = 3
        store.collection.update(ids=[n.node_id], metadatas=[meta])
    store._migrate_layers()
    assert store.get("a").superseded_by == ["b"]
    assert not store.get("a").is_latest and store.get("b").is_latest
    assert store.get("a").to_metadata()["layer_schema_version"] == 4


def test_large_retirement_chunks_and_recovers_partial_patch_failure(store, monkeypatch):
    store.upsert([node("a")])
    store.upsert(
        [
            replace(node(f"d{i}"), layer=Layer.L3_DERIVED, custom={"source_ids": ["a"]})
            for i in range(5)
        ]
    )
    original = store.collection.update
    calls = []
    fail = True
    monkeypatch.setattr(store.client, "get_max_batch_size", lambda: 2)

    def update(**kw):
        nonlocal fail
        calls.append(len(kw["ids"]))
        assert len(kw["ids"]) <= 2
        if len(calls) == 2 and fail:
            fail = False
            raise OSError("crash in second chunk")
        return original(**kw)

    monkeypatch.setattr(store.collection, "update", update)
    with pytest.raises(OSError):
        apply(store, [node("b", supersedes=["a"])], {"a": 0})
    assert store.get("b").is_latest
    assert all(not store.get(f"d{i}").is_latest for i in range(5))
    assert not store.get("a").is_latest and len(calls) >= 5
    assert (
        store._journal.execute("SELECT count(*) FROM plans WHERE done=0").fetchone()[0]
        == 0
    )
