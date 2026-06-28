# -*- coding: utf-8 -*-
"""Coding memory reconciler — decide ADD/UPDATE/SKIP for new drafts.

Simple task-similarity check: if a draft's task is very similar to an existing
memory's task, UPDATE; otherwise ADD. No LLM needed (keeps it fast).
"""
import logging
from typing import List, Optional

from dual_mem.coding.types import CodingMemoryDraft, CodingMemory, CodingReconcileOp

logger = logging.getLogger("dual_mem.coding.reconciler")

_TASK_SIMILARITY_THRESHOLD = 0.85


class CodingMemoryReconciler:
    """Rule-based reconciler — no extra LLM call."""

    def reconcile(
        self,
        drafts: List[CodingMemoryDraft],
        existing: List[CodingMemory],
    ) -> List[CodingReconcileOp]:
        """Decide ops for each draft based on task similarity to existing."""
        ops: List[CodingReconcileOp] = []
        for i, draft in enumerate(drafts):
            match = self._find_match(draft, existing)
            if match:
                ops.append(CodingReconcileOp(
                    action="UPDATE",
                    draft_idx=i,
                    target_memory_id=match.memory_id,
                    reason=f"task similar to existing: {match.task[:60]}",
                ))
            else:
                ops.append(CodingReconcileOp(
                    action="ADD",
                    draft_idx=i,
                    reason="new coding memory",
                ))
        logger.info("[coding-reconcile] %d drafts → %d ADD, %d UPDATE",
                    len(drafts),
                    sum(1 for o in ops if o.action == "ADD"),
                    sum(1 for o in ops if o.action == "UPDATE"))
        return ops

    @staticmethod
    def _find_match(
        draft: CodingMemoryDraft, existing: List[CodingMemory]
    ) -> Optional[CodingMemory]:
        """Find an existing memory with high task overlap."""
        draft_words = set(draft.task.lower().split())
        if not draft_words:
            return None
        best = None
        best_score = 0.0
        for mem in existing:
            mem_words = set(mem.task.lower().split())
            if not mem_words:
                continue
            overlap = len(draft_words & mem_words)
            score = overlap / max(len(draft_words | mem_words), 1)
            if score > best_score:
                best_score = score
                best = mem
        if best and best_score >= _TASK_SIMILARITY_THRESHOLD:
            return best
        return None
