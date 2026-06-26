# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Graph expansion for the hybrid read path: from a list of anchors, surface their 1-hop neighbours.
For schema anchors we follow DERIVED_FROM to their evidence facts. For fact / identity
anchors we add timeline neighbours (same session_id ± nearby gmt_created) and same-session
summaries. Returns the merged neighbour list with attenuated scores so the fusion stage can
rank them against the original anchors.
"""
import logging
from dataclasses import dataclass

from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.anchor_search import AnchorNode
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.retrieval.expand")

# Multiplicative attenuation applied to the source anchor score so neighbours rank below
# their seed but still compete with weaker direct hits.
_EVIDENCE_ATTN = 0.7
_SESSION_ATTN = 0.6
_TIMELINE_ATTN = 0.5

# Cap how far we expand to keep latency bounded.
_MAX_SCHEMA_SOURCES = 5
_MAX_EVIDENCE_PER_SCHEMA = 8
_MAX_FACT_SOURCES = 10


@dataclass
class ExpansionResult:
    """Neighbours surfaced by graph expansion, ready to be merged into anchor list."""

    expanded: list[AnchorNode]
    edge_counts: dict[str, int]


class GraphExpander:
    """Expand anchor seeds 1-hop using the graph store (dual) + vector store metadata."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    def expand(
        self,
        *,
        anchors: list[AnchorNode],
        app_ids: list[str],
        user_id: str,
    ) -> ExpansionResult:
        """Run all 1-hop expansions; return neighbours with their source-edge counts."""
        if not anchors:
            return ExpansionResult(expanded=[], edge_counts={})

        seen_ids: set[str] = {a.node_id for a in anchors}
        expanded: list[AnchorNode] = []
        edge_counts: dict[str, int] = {"evidence": 0, "session": 0, "timeline": 0}

        # Schema → evidence (dual only).
        schema_seeds = [a for a in anchors if a.node.layer is Layer.L6_SCHEMA][:_MAX_SCHEMA_SOURCES]
        for seed in schema_seeds:
            for fact in self._evidence_of(seed.node, app_ids=app_ids, user_id=user_id):
                if fact.node_id in seen_ids:
                    continue
                seen_ids.add(fact.node_id)
                expanded.append(
                    AnchorNode(
                        node=fact,
                        score=seed.score * _EVIDENCE_ATTN,
                        source_path="expand_evidence",
                    )
                )
                edge_counts["evidence"] += 1

        # Fact / identity seeds → timeline neighbours (same session, nearby gmt_created).
        fact_seeds = [
            a
            for a in anchors
            if a.node.layer in (Layer.L2_FACT, Layer.L4_IDENTITY) and a.node.session_id
        ][:_MAX_FACT_SOURCES]
        for seed in fact_seeds:
            for neighbour, kind in self._session_timeline(
                seed.node, app_ids=app_ids, user_id=user_id
            ):
                if neighbour.node_id in seen_ids:
                    continue
                seen_ids.add(neighbour.node_id)
                attn = _SESSION_ATTN if kind == "session" else _TIMELINE_ATTN
                expanded.append(
                    AnchorNode(
                        node=neighbour,
                        score=seed.score * attn,
                        source_path=f"expand_{kind}",
                    )
                )
                edge_counts[kind] += 1

        logger.debug(
            "expand anchors=%d expanded=%d edges=%s",
            len(anchors), len(expanded), edge_counts,
        )
        return ExpansionResult(expanded=expanded, edge_counts=edge_counts)

    # ---- 1-hop helpers ---------------------------------------------------------------

    def _evidence_of(
        self, schema_node: MemoryNode, *, app_ids: list[str], user_id: str
    ) -> list[MemoryNode]:
        """Walk DERIVED_FROM edges from a schema node to its evidence facts (dual only)."""
        graph = self.factory.graph
        if graph is None:
            return []
        try:
            evidence_ids = graph.evidence_of(schema_node.node_id)
        except Exception:
            return []
        out: list[MemoryNode] = []
        for fid in evidence_ids[:_MAX_EVIDENCE_PER_SCHEMA]:
            node = self.factory.vector.get(fid)
            if node is None or node.status is not MemoryStatus.ACTIVE:
                continue
            if node.user_id != user_id or node.app_id not in app_ids:
                continue
            out.append(node)
        return out

    def _session_timeline(
        self, seed: MemoryNode, *, app_ids: list[str], user_id: str
    ) -> list[tuple[MemoryNode, str]]:
        """Find same-session L3 summaries + nearby fact/identity siblings via gmt_created proximity."""
        out: list[tuple[MemoryNode, str]] = []

        # Same-session summary.
        if seed.session_id:
            where_summary = build_filter(
                app_ids=app_ids,
                user_id=user_id,
                session_ids=[seed.session_id],
                layers=[Layer.L3_SUMMARY],
                statuses=[MemoryStatus.ACTIVE],
            )
            try:
                summaries = self.factory.vector.get_many(where_summary, limit=2)
            except Exception:
                summaries = []
            for s in summaries:
                if s.node_id == seed.node_id:
                    continue
                out.append((s, "session"))

        # Same-session timeline siblings (other facts in the same session).
        if seed.session_id:
            where_timeline = build_filter(
                app_ids=app_ids,
                user_id=user_id,
                session_ids=[seed.session_id],
                layers=[Layer.L2_FACT, Layer.L4_IDENTITY],
                statuses=[MemoryStatus.ACTIVE],
            )
            try:
                siblings = self.factory.vector.get_many(where_timeline, limit=8)
            except Exception:
                siblings = []
            for sib in siblings:
                if sib.node_id == seed.node_id:
                    continue
                out.append((sib, "timeline"))

        return out
