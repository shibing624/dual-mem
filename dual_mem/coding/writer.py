"""Retain full L0 evidence, generate candidates, decide with JEV, append versions."""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from dual_mem.coding.extractor import CodingMemoryExtractor
from dual_mem.coding.judge import classify_messages_is_coding
from dual_mem.coding.preproc import has_any_tool_message
from dual_mem.coding.reconciler import CodingMemoryReconciler
from dual_mem.coding.store import CodingMemoryStore
from dual_mem.coding.types import CodingDecisionError, CodingMemory
from dual_mem.errors import IdempotencyConflictError
from dual_mem.locks import LockRegistry
from dual_mem.providers.jev import DecisionProvider

_RETIRES = {"OVERRIDE", "MERGE", "NEGATE"}


class CodingWriter:
    def __init__(
        self,
        *,
        store: CodingMemoryStore,
        extractor: CodingMemoryExtractor,
        reconciler: CodingMemoryReconciler,
        jev: DecisionProvider,
    ) -> None:
        self.store = store
        self.extractor = extractor
        self.reconciler = reconciler
        self.jev = jev
        self._scope_locks = LockRegistry()

    async def write(
        self,
        messages: list[dict],
        *,
        user_id: str,
        agent_id: str = "",
        app_id: str = "default",
        session_id: str | None = None,
        workspace_id: str | None = None,
        branch: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return None for non-coding; propagate failures with retained L0 evidence.

        Successful result includes raw_id/source_ids and decisions. RAW_ONLY is
        explicit source retention, not a claim that candidates were persisted.

        With ``request_id`` the write is idempotent: the raw source id and memory ids
        are derived from it, the result is recorded in the publishing transaction,
        and a repeat returns that result without new decisions. Reusing the id with
        different input raises ``IdempotencyConflictError``.
        """
        if not has_any_tool_message(messages):
            return None
        scope = (app_id, user_id, agent_id)
        request_key = None
        raw_id = None
        fingerprint = ""
        if request_id is not None:
            fingerprint = hashlib.sha256(
                json.dumps(
                    [messages, session_id, workspace_id, branch], ensure_ascii=False
                ).encode()
            ).hexdigest()
            request_key = f"coding:{json.dumps([*scope, request_id])}"
            saved = self.store.request_result(request_key)
            if saved is not None:
                if saved["fingerprint"] != fingerprint:
                    raise IdempotencyConflictError(
                        "request_id reused with different input"
                    )
                return saved["result"]
            raw_id = str(uuid.uuid5(uuid.NAMESPACE_URL, request_key + ":raw"))
        # Retain even if scene classification fails, before any semantic service.
        raw_id = await self.store.retain_source(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            app_id=app_id,
            session_id=session_id,
            workspace_id=workspace_id,
            branch=branch,
            raw_id=raw_id,
        )
        if not await classify_messages_is_coding(messages, self.jev):
            # Not coding is re-decided on retry; it publishes nothing.
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
        # Extraction overlaps freely; snapshot → decide → publish is serialized per
        # scope so a concurrent write sees the previous one's heads instead of
        # failing the epoch check.
        async with self._scope_locks.get(json.dumps(scope)):
            epoch, existing, histories = self.store.snapshot(
                user_id=user_id, agent_id=agent_id, app_id=app_id
            )
            ops = await self.reconciler.reconcile(
                drafts,
                existing,
                source_messages=messages,
                history_versions=histories,
            )
            now = datetime.now(timezone.utc)
            memories: list[CodingMemory] = []
            archives: list[str] = []
            decisions: list[dict[str, Any]] = []
            for op in ops:
                draft = drafts[op.draft_idx]
                if op.action == "ADD":
                    memories.append(
                        CodingMemory(
                            memory_id=(
                                str(
                                    uuid.uuid5(
                                        uuid.NAMESPACE_URL,
                                        f"{request_key}:{op.draft_idx}",
                                    )
                                )
                                if request_key
                                else str(uuid.uuid4())
                            ),
                            **draft.to_dict(),
                            created_at=now,
                            updated_at=now,
                            supersedes=(
                                [op.target_memory_id]
                                if op.update_type in _RETIRES
                                else []
                            ),
                            source_ids=[raw_id],
                        )
                    )
                elif op.action == "DELETE":
                    archives.append(op.target_memory_id)
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
            if not drafts:
                decisions.append(
                    {
                        "action": "RAW_ONLY",
                        "source_ids": [raw_id],
                        "reason": "RAW_ONLY: no explicit candidates",
                    }
                )
            memory_ids = [m.memory_id for m in memories]
            replaced = {old for m in memories for old in m.supersedes}
            if replaced.intersection(archives):
                # Contradictory decisions: fail before reserving any row or vector.
                raise CodingDecisionError(
                    "coding head is both replaced and archived in one batch"
                )
            result = {
                "success": True,
                "raw_id": raw_id,
                "source_ids": [raw_id],
                "memory_ids": memory_ids,
                # Same contract as the chat path: a version was appended or archived.
                "commit_passed": bool(memories or archives),
                "decisions": decisions,
            }
            if not memories and not archives:
                # Another process sharing the database may have moved the heads.
                self.store.assert_epoch(scope, epoch)
            record = None
            if request_key is not None:
                # A crashed earlier attempt may have reserved these IDs unpublished.
                self.store.discard_unpublished(memory_ids)
                record = (request_key, {"fingerprint": fingerprint, "result": result})
            await self.store.add_many(
                memories,
                expected_epoch=epoch,
                archive_ids=archives,
                request=record,
            )
        return result

    async def search(
        self,
        *,
        query: str,
        user_id: str,
        agent_id: str = "",
        app_id: str = "default",
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        return await self.store.search(
            query=query, user_id=user_id, agent_id=agent_id, app_id=app_id, top_k=top_k
        )
