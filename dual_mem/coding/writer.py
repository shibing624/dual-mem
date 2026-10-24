"""Procedure pipeline: JEV scene → candidates → JEV decisions → version plan.

Procedures share the main vector store, the L0_RAW source and the request journal
with chat memories. The chat facts of the same write and its procedures therefore
cite the same raw node (provenance), retire through the same version plan, and
feed the same System 2 digest.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass, field, replace

from dual_mem.coding.extractor import CodingMemoryExtractor
from dual_mem.coding.judge import classify_messages_is_coding
from dual_mem.coding.preproc import clip_tool_results
from dual_mem.coding.reconciler import CodingMemoryReconciler
from dual_mem.coding.types import (
    PROCEDURE_KIND,
    RETIRING_UPDATES,
    CodingMemory,
    CodingMemoryDraft,
    CodingReconcileOp,
)
from dual_mem.errors import StaleDecisionError
from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.sdk_models import MemoryDecision
from dual_mem.types import Layer, MemoryNode, MemoryStatus


@dataclass
class ProcedureOutcome:
    """Result of the procedure half of one write."""

    # False when JEV judged the trajectory not to be engineering work.
    coding: bool
    node_ids: list[str] = field(default_factory=list)
    decisions: list[MemoryDecision] = field(default_factory=list)
    committed: bool = False


def _settle(ops: list[CodingReconcileOp]) -> list[CodingReconcileOp]:
    """Give each current head at most one retiring claim, without another model call.

    A version plan retires a head once: a replacement and an archive of the same
    head, or two of either, contradict each other. The highest-confidence claim
    wins; a losing replacement is kept as an independent version and a losing
    archive keeps only its raw source.
    """
    out = list(ops)
    claims: dict[str, list[int]] = {}
    for i, op in enumerate(ops):
        retiring = op.action == "DELETE" or (
            op.action == "ADD" and op.update_type in RETIRING_UPDATES
        )
        if retiring and op.target_memory_id:
            claims.setdefault(op.target_memory_id, []).append(i)
    for idxs in claims.values():
        best = max(idxs, key=lambda i: ops[i].confidence)
        for i in idxs:
            if i == best:
                continue
            if ops[i].action == "ADD":
                out[i] = replace(ops[i], update_type="SUPPLEMENT", target_memory_id=None)
            else:
                out[i] = replace(
                    ops[i],
                    action="RAW_ONLY",
                    target_memory_id=None,
                    reason=f"{ops[i].reason}: head already claimed in this write",
                )
    return out


class ProcedureWriter:
    def __init__(self, *, factory: ComponentFactory) -> None:
        self.factory = factory
        settings = factory.settings
        self.extractor = CodingMemoryExtractor(llm=factory.llm)
        self.reconciler = CodingMemoryReconciler(
            jev=factory.jev, candidate_limit=min(settings.reconcile_candidate_limit, 253)
        )
        self.tool_result_max_chars = settings.procedure_tool_result_max_chars

    @staticmethod
    def request_key(raw: MemoryNode, request_id: str) -> str:
        return f"procedure:{raw.node_id}:{request_id}"

    def _scope_nodes(self, scope: tuple[str, str, str]) -> list[MemoryNode]:
        where = build_filter(
            app_ids=[scope[0]],
            user_id=scope[1],
            agent_ids=[scope[2]],
            layers=[Layer.L1_FACT],
            statuses=[MemoryStatus.ACTIVE],
        )
        where["kind"] = PROCEDURE_KIND
        return self.factory.vector.get_many(where, limit=None)

    def snapshot(
        self, scope: tuple[str, str, str]
    ) -> tuple[list[CodingMemory], dict[str, list[dict]]]:
        """Current heads of the scope and, per head, every ancestor version."""
        nodes = {n.node_id: CodingMemory.from_node(n) for n in self._scope_nodes(scope)}
        heads = [m for m in nodes.values() if m.is_latest]
        histories: dict[str, list[dict]] = {}
        for head in heads:
            seen: dict[str, CodingMemory] = {}
            pending = list(head.supersedes)
            while pending:
                old_id = pending.pop()
                if old_id in seen or old_id not in nodes:
                    continue
                seen[old_id] = nodes[old_id]
                pending.extend(nodes[old_id].supersedes)
            if seen:
                histories[head.memory_id] = [m.to_dict() for m in seen.values()]
        heads.sort(key=lambda m: (m.created_at, m.memory_id), reverse=True)
        return heads, histories

    async def write(
        self,
        *,
        messages: list[dict],
        raw: MemoryNode,
        request_id: str,
        workspace_id: str | None = None,
        branch: str | None = None,
    ) -> ProcedureOutcome:
        """Decide and commit the procedures of one trajectory.

        A committed request replays from the plan journal. Model failures
        propagate; the trajectory is already retained as ``raw``.
        """
        request_key = self.request_key(raw, request_id)
        done = self.factory.vector.completed_request(request_key)
        if done is not None:
            return ProcedureOutcome(
                coding=True,
                node_ids=list(done["extra_node_ids"]),
                decisions=[MemoryDecision(**d) for d in done["decisions"]],
                committed=done["commit_passed"],
            )
        source = clip_tool_results(messages, self.tool_result_max_chars)
        if not await classify_messages_is_coding(source, self.factory.jev):
            # Not coding is re-decided on retry; nothing is recorded.
            return ProcedureOutcome(coding=False)
        drafts = await self.extractor.extract(
            source,
            workspace_id=workspace_id,
            branch=branch,
            session_id=raw.session_id,
        )
        scope = (raw.app_id, raw.user_id, raw.agent_id)
        # Same per-scope decision lock as the chat path: a concurrent write sees
        # the heads this one commits instead of planning against stale ones.
        async with self.factory.commit_locks.get("decision::" + "::".join(scope)):
            heads, histories = await asyncio.to_thread(self.snapshot, scope)
            ops = await self.reconciler.reconcile(
                drafts, heads, source_messages=source, history_versions=histories
            )
            return await self._commit(
                _settle(ops), drafts, heads, raw, scope, request_key
            )

    async def _commit(
        self,
        ops: list[CodingReconcileOp],
        drafts: list[CodingMemoryDraft],
        heads: list[CodingMemory],
        raw: MemoryNode,
        scope: tuple[str, str, str],
        request_key: str,
    ) -> ProcedureOutcome:
        revisions = {m.memory_id: m.revision for m in heads}
        adds = [op for op in ops if op.action == "ADD"]
        embeddings = (
            await self.factory.embed.embed_batch(
                [drafts[op.draft_idx].index_text() for op in adds]
            )
            if adds
            else []
        )
        nodes: list[MemoryNode] = []
        archives: list[str] = []
        expected: dict[str, int] = {}
        decisions: list[MemoryDecision] = []
        for op in ops:
            draft = drafts[op.draft_idx]
            proposal = f"procedure:{op.draft_idx}"
            target = op.target_memory_id
            node_id = ""
            if op.action == "ADD":
                retires = op.update_type in RETIRING_UPDATES
                if retires:
                    expected[target] = revisions[target]
                node_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"{request_key}:{op.draft_idx}")
                )
                nodes.append(
                    MemoryNode(
                        node_id=node_id,
                        content=draft.content(),
                        layer=Layer.L1_FACT,
                        kind=PROCEDURE_KIND,
                        app_id=raw.app_id,
                        user_id=raw.user_id,
                        agent_id=raw.agent_id,
                        session_id=raw.session_id,
                        memory_at=raw.memory_at,
                        tags=list(draft.files),
                        supersedes=[target] if retires else [],
                        embedding=embeddings[adds.index(op)],
                        custom={
                            # Stored whole (None kept): the reader requires every field.
                            "procedure": asdict(draft.detail()),
                            "source_ids": [raw.node_id],
                            "source_node_id": raw.node_id,
                            "update_type": op.update_type,
                            "conflicts_with": (
                                [target] if op.update_type == "CONFLICT" else []
                            ),
                            "proposal_ids": [proposal],
                        },
                    )
                )
            elif op.action == "DELETE":
                expected[target] = revisions[target]
                archives.append(target)
            decisions.append(
                MemoryDecision(
                    action=op.action,
                    update_type=op.update_type or None,
                    targets=[target] if target else [],
                    proposal_ids=[node_id or proposal],
                    source_ids=[raw.node_id],
                    reason=op.reason,
                    confidence=op.confidence,
                )
            )
        if not drafts:
            decisions.append(
                MemoryDecision(
                    action="RAW_ONLY",
                    source_ids=[raw.node_id],
                    reason="RAW_ONLY: no explicit candidates",
                )
            )
        ok = await asyncio.to_thread(
            self.factory.vector.apply_version_plan,
            nodes=nodes,
            expected=expected,
            archives=archives,
            scope=scope,
            request_key=request_key,
            decisions=[d.to_dict() for d in decisions],
        )
        if not ok:
            raise StaleDecisionError("procedure targets changed concurrently; retry write")
        return ProcedureOutcome(
            coding=True,
            node_ids=[n.node_id for n in nodes],
            decisions=decisions,
            committed=bool(nodes or archives),
        )
