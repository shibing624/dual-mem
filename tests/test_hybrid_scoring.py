# -*- coding: utf-8 -*-
"""Unit tests for the hybrid fusion helpers (evidence boost + semantic/BM25 fusion)."""
from dual_mem.retrieval.hybrid_scoring import compute_evidence_boost, score_vdb_node


def test_evidence_boost_zero_when_no_evidence():
    assert compute_evidence_boost(0, saturate=5, max_boost=0.3) == 0.0
    assert compute_evidence_boost(3, saturate=0, max_boost=0.3) == 0.0


def test_evidence_boost_saturates_at_max():
    # 5/5 == 1.0 of the cap; anything beyond saturate is clamped.
    assert compute_evidence_boost(5, saturate=5, max_boost=0.3) == 0.3
    assert compute_evidence_boost(20, saturate=5, max_boost=0.3) == 0.3
    # Partial support is a linear fraction of the cap.
    assert compute_evidence_boost(2, saturate=5, max_boost=0.3) == 0.3 * (2 / 5)


def test_score_vdb_fusion_is_weighted_sum():
    assert score_vdb_node(1.0, 0.0, 0.6, 0.4) == 0.6
    assert score_vdb_node(0.0, 1.0, 0.6, 0.4) == 0.4
    assert score_vdb_node(0.5, 0.5, 0.6, 0.4) == 0.5


def test_fuse_then_gate_rescues_high_bm25_low_semantic():
    """I3 invariant: a weak-semantic but strong-keyword hit clears a 0.4 gate via fusion.

    The raw semantic score (0.3) would be dropped by a pre-fusion min_score=0.4 filter, but
    the fused score (0.3*0.6 + 1.0*0.4 = 0.58) survives — this is exactly why the engine
    fuses first and gates on the fused score.
    """
    raw_semantic = 0.3
    fused = score_vdb_node(raw_semantic, 1.0, 0.6, 0.4)
    gate = 0.4
    assert raw_semantic < gate  # old pre-filter would have dropped it
    assert fused >= gate  # fuse-then-gate keeps it
