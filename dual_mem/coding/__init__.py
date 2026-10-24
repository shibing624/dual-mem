# -*- coding: utf-8 -*-
"""Coding scene: procedure memories from coding-agent tool trajectories.

A procedure is an L1_FACT node with ``kind="procedure"`` in the main store. It
shares the raw source, version chain, request journal and System 2 digest with
chat memories; only its candidate generation and JEV decision stream differ.

Flow inside ``MemoryClient.add(messages=...)``: a tool trace (structural check)
→ JEV scene question → procedure candidates → JEV decisions → version plan,
alongside the unchanged chat extraction of the same conversation.
"""
from dual_mem.coding.types import (
    BOUNDARY_SCOPES,
    PROCEDURE_KIND,
    CodingDecisionError,
    CodingMemory,
    CodingMemoryDraft,
    CodingReconcileOp,
)

__all__ = [
    "BOUNDARY_SCOPES",
    "PROCEDURE_KIND",
    "CodingDecisionError",
    "CodingMemory",
    "CodingMemoryDraft",
    "CodingReconcileOp",
]
