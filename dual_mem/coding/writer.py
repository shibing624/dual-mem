# -*- coding: utf-8 -*-
"""Coding memory writer — orchestrates judge → preproc → extract → reconcile → store.

Flow:
    has_any_tool_message(messages)?
      ├── False → return None (caller should use chat path)
      └── True → judge: is_coding?
                   ├── False → return None (caller strips tool msgs, uses chat path)
                   └── True → extract → reconcile → store → return result
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from dual_mem.providers.llm import LLMClient
from dual_mem.providers.embedding import EmbedService
from dual_mem.storage.vector_store import VectorStore
from dual_mem.coding.extractor import CodingMemoryExtractor
from dual_mem.coding.judge import classify_messages_is_coding
from dual_mem.coding.preproc import has_any_tool_message, truncate_messages
from dual_mem.coding.reconciler import CodingMemoryReconciler
from dual_mem.coding.store import CodingMemoryStore
from dual_mem.coding.types import CodingMemory

logger = logging.getLogger("dual_mem.coding.writer")


class CodingWriter:
    """Coding write-path orchestrator."""

    def __init__(
        self,
        *,
        store: CodingMemoryStore,
        extractor: CodingMemoryExtractor,
        reconciler: CodingMemoryReconciler,
        llm: LLMClient,
    ):
        self.store = store
        self.extractor = extractor
        self.reconciler = reconciler
        self.llm = llm

    async def write(
        self,
        messages: List[dict],
        *,
        user_id: str,
        agent_id: str = "default_agent",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run coding write path. Returns None if messages are not coding."""
        if not has_any_tool_message(messages):
            return None

        is_coding = await classify_messages_is_coding(messages, self.llm)
        if not is_coding:
            return None

        drafts = await self.extractor.extract(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            branch=branch,
            session_id=session_id,
        )
        if not drafts:
            return {"success": True, "scene": "coding", "memory_ids": [], "ops": []}

        existing = self.store.list_by_user(user_id=user_id, agent_id=agent_id)
        ops = self.reconciler.reconcile(drafts, existing)

        memory_ids: List[str] = []
        for op in ops:
            if op.draft_idx is None or op.draft_idx >= len(drafts):
                continue
            draft = drafts[op.draft_idx]
            now = datetime.now()

            if op.action == "ADD":
                mem = CodingMemory(
                    memory_id=str(uuid.uuid4()),
                    user_id=draft.user_id,
                    agent_id=draft.agent_id,
                    task=draft.task,
                    search_keys=draft.search_keys,
                    solution=draft.solution,
                    boundary_envs=draft.boundary_envs,
                    boundary_scope=draft.boundary_scope,
                    workspace_id=draft.workspace_id,
                    branch=draft.branch,
                    session_id=draft.session_id,
                    files=draft.files,
                    confidence=draft.confidence,
                    source=draft.source,
                    created_at=now,
                    updated_at=now,
                )
                await self.store.add(mem)
                memory_ids.append(mem.memory_id)

            elif op.action == "UPDATE" and op.target_memory_id:
                old = self.store.get(op.target_memory_id)
                if old:
                    old.task = draft.task
                    old.search_keys = draft.search_keys
                    old.solution = draft.solution
                    old.boundary_envs = draft.boundary_envs
                    old.boundary_scope = draft.boundary_scope
                    old.files = draft.files
                    old.confidence = draft.confidence
                    old.updated_at = now
                    await self.store.update(old)
                    memory_ids.append(old.memory_id)

        logger.info("[coding-write] user=%s drafts=%d stored=%d", user_id, len(drafts), len(memory_ids))
        return {
            "success": True,
            "scene": "coding",
            "memory_ids": memory_ids,
            "ops": [o.__dict__ for o in ops],
        }

    async def search(
        self, *, query: str, user_id: str, agent_id: str = "default_agent", top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search coding memories."""
        return await self.store.search(query=query, user_id=user_id, agent_id=agent_id, top_k=top_k)
