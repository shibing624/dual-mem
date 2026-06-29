# -*- coding: utf-8 -*-
"""Coding/tool-use memory subsystem — separate write/store/search path for
engineering conversations that contain tool calls (Read/Edit/Bash/etc).

Flow: add() → has_tool_messages? → judge(is_coding?) → coding writer
      (extract → reconcile → coding store) OR chat path (normal extract).
"""
from dual_mem.coding.types import (
    CodingMemoryDraft,
    CodingMemory,
    CodingReconcileOp,
    BOUNDARY_SCOPES,
)
from dual_mem.coding.writer import CodingWriter
from dual_mem.coding.store import CodingMemoryStore
from dual_mem.coding.extractor import CodingMemoryExtractor
from dual_mem.coding.reconciler import CodingMemoryReconciler
from dual_mem.coding.judge import classify_messages_is_coding
from dual_mem.coding.preproc import (
    has_any_tool_message,
    strip_tool_messages,
    truncate_messages,
    extract_files,
    extract_tool_summary,
)

__all__ = [
    "CodingWriter",
    "CodingMemoryStore",
    "CodingMemoryExtractor",
    "CodingMemoryReconciler",
    "CodingMemoryDraft",
    "CodingMemory",
    "CodingReconcileOp",
    "BOUNDARY_SCOPES",
    "classify_messages_is_coding",
    "has_any_tool_message",
    "strip_tool_messages",
    "truncate_messages",
    "extract_files",
    "extract_tool_summary",
]
