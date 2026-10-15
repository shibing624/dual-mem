from dual_mem.agent.reconciler import (
    collect_merged_timestamps,
    reflect_custom,
    ReconcileOp,
)
from dual_mem.types import MemoryNode, Layer


def test_merge_preserves_every_event_timestamp():
    old = MemoryNode(
        content="80 yuan",
        layer=Layer.L1_FACT,
        app_id="a",
        user_id="u",
        memory_at=10,
        custom={"merged_timestamps": [1, 2, 10]},
    )
    assert collect_merged_timestamps(
        update_type="MERGE", new_memory_at=20, old_nodes=[old], extra=[2, 30]
    ) == [1, 2, 10, 20, 30]
    assert (
        collect_merged_timestamps(
            update_type="OVERRIDE", new_memory_at=20, old_nodes=[old]
        )
        == []
    )


def test_conflict_and_temporal_metadata_survive_decision_serialization():
    op = ReconcileOp(
        update_type="CONFLICT",
        conflicts_with=["old"],
        source_ids=["raw"],
        proposal_ids=["p0"],
    )
    custom = reflect_custom(op)
    assert custom["conflicts_with"] == ["old"] and custom["source_ids"] == ["raw"]
    temporal = reflect_custom(
        ReconcileOp(update_type="TEMPORAL", temporal_scope="2026-10-15")
    )
    assert temporal["temporal_scope"] == "2026-10-15"
