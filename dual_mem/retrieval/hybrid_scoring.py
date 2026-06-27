# -*- coding: utf-8 -*-
"""Hybrid scoring helpers (graph evidence boost, semantic + BM25 fusion)."""


def compute_evidence_boost(
    evidence_count: int,
    saturate: int = 5,
    max_boost: float = 0.3,
) -> float:
    """Linear evidence boost for graph L6 intra-pool ranking."""
    if evidence_count <= 0 or saturate <= 0:
        return 0.0
    return min(evidence_count / float(saturate), 1.0) * max_boost


def score_vdb_node(
    semantic_score: float,
    bm25_score: float,
    w_sem: float = 0.6,
    w_bm25: float = 0.4,
) -> float:
    """Fuse semantic + BM25 into [0, 1]."""
    return semantic_score * w_sem + bm25_score * w_bm25
