"""Retain full L0 evidence, generate candidates, decide with JEV, append versions."""

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from dual_mem.coding.extractor import CodingMemoryExtractor
from dual_mem.coding.judge import classify_messages_is_coding
from dual_mem.coding.preproc import has_any_tool_message
from dual_mem.coding.reconciler import CodingMemoryReconciler
from dual_mem.coding.store import CodingMemoryStore
from dual_mem.coding.types import CodingDecisionError, CodingMemory
from dual_mem.providers.jev import DecisionProvider
from dual_mem.providers.llm import LLMClient


class CodingWriter:
    def __init__(
        self,
        *,
        store: CodingMemoryStore,
        extractor: CodingMemoryExtractor,
        reconciler: CodingMemoryReconciler,
        llm: LLMClient,
        jev: DecisionProvider,
    ) -> None:
        self.store = store
        self.extractor = extractor
        self.reconciler = reconciler
        self.llm = llm
        self.jev = jev

    async def write(
        self,
        messages: list[dict],
        *,
        user_id: str,
        agent_id: str = "default_agent",
        app_id: str = "default",
        session_id: str | None = None,
        workspace_id: str | None = None,
        branch: str | None = None,
    ) -> dict[str, Any] | None:
        """Return None for non-coding; propagate failures with retained L0 evidence.

        Successful result includes raw_id/source_ids and decisions. RAW_ONLY is
        explicit source retention, not a claim that candidates were persisted.
        """
        if not has_any_tool_message(messages):
            return None
        # Retain even if scene classification fails, before any semantic service.
        raw_id = await self.store.retain_source(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            session_id=session_id,
            workspace_id=workspace_id,
            branch=branch,
        )
        if not await classify_messages_is_coding(messages, self.jev):
            return None
        drafts = await self.extractor.extract(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            workspace_id=workspace_id,
            branch=branch,
            session_id=session_id,
        )
        epoch, existing, histories = self.store.snapshot(
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
        )
        ops = await self.reconciler.reconcile(
            drafts,
            existing,
            source_messages=messages,
            history_versions=histories,
        )
        if len(ops) != len(drafts) or {op.draft_idx for op in ops} != set(
            range(len(drafts))
        ):
            raise CodingDecisionError(
                "coding operations must cover every draft exactly"
            )
        memories = []
        archives: list[str] = []
        decisions = []
        retires = {"OVERRIDE", "MERGE", "NEGATE"}
        for op in ops:
            if op.action not in ("ADD", "SKIP", "DELETE", "RAW_ONLY"):
                raise CodingDecisionError("unsupported coding write action")
            draft = drafts[op.draft_idx]
            if (draft.app_id, draft.user_id, draft.agent_id) != (
                app_id,
                user_id,
                agent_id,
            ):
                raise CodingDecisionError("coding candidate crosses request scope")
            if op.action == "ADD":
                needs_target = op.update_type in {"OVERRIDE", "MERGE", "NEGATE", "CONFLICT"}
                if needs_target != bool(op.target_memory_id):
                    raise CodingDecisionError("coding write action/target mismatch")
                now = datetime.now(timezone.utc)
                mem = CodingMemory(
                    memory_id=str(uuid.uuid4()),
                    **draft.to_dict(),
                    created_at=now,
                    updated_at=now,
                    supersedes=(
                        [op.target_memory_id] if op.update_type in retires else []
                    ),
                    source_ids=[raw_id],
                )
                memories.append(mem)
            elif op.action == "DELETE":
                if not op.target_memory_id:
                    raise CodingDecisionError("coding DELETE requires a current target")
                archives.append(op.target_memory_id)
            elif op.action == "SKIP":
                if not op.target_memory_id:
                    raise CodingDecisionError("coding SKIP requires a current target")
            decisions.append(
                {
                    "action": op.action,
                    "update_type": op.update_type or None,
                    "targets": [op.target_memory_id] if op.target_memory_id else [],
                    "proposal_ids": [f"p{op.draft_idx}"],
                    "source_ids": [raw_id],
                    "reason": op.reason,
                }
            )
        memory_ids = await self.store.add_many(
            memories, expected_epoch=epoch, archive_ids=archives
        )
        if not memories and not archives:
            self.store.assert_epoch((app_id, user_id, agent_id), epoch)
        if not drafts:
            decisions.append(
                {
                    "action": "RAW_ONLY",
                    "source_ids": [raw_id],
                    "reason": "JEV confirmed no durable information",
                }
            )
        return {
            "success": True,
            "scene": "coding",
            "raw_id": raw_id,
            "source_ids": [raw_id],
            "memory_ids": memory_ids,
            "extra_node_ids": [f"{mid}_key0" for mid in memory_ids],
            "commit_passed": any(
                op.action in ("ADD", "SKIP", "DELETE") for op in ops
            ),
            "decisions": decisions,
            "ops": [asdict(op) for op in ops],
        }

    async def search(
        self,
        *,
        query: str,
        user_id: str,
        agent_id: str = "default_agent",
        app_id: str = "default",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        return await self.store.search(
            query=query, user_id=user_id, agent_id=agent_id, app_id=app_id, top_k=top_k
        )
