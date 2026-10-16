"""Bounded cross-evidence generation, Jev validation and versioned cognition.

Every fresh fact enters a real input batch; batches are ordered by topic so
related evidence is reasoned over together. Only IDs actually received and
accepted by both models are acknowledged. A rejected batch is bisected within
the same digest so the remaining evidence still commits; a single fact that is
rejected on ``s2_max_attempts`` digests is quarantined and stops blocking its scope.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone

import numpy as np

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.errors import DecisionContextError, StaleDecisionError
from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.system2.agent")

_KINDS = ("schema", "summary", "intention", "preference")
_SYSTEM = (
    "Generate cross-evidence memories, not relation decisions. Use only supplied fact IDs. "
    'Return {"memories":[{"content":...,"kind":"schema|summary|intention|preference","evidence_ids":[...]}],'
    '"processed_ids":[...]}. List every input ID actually examined, even when no useful inference is warranted. '
    "kind=preference is a user preference, taste, habit or value that the cited facts imply but do not state "
    "outright (e.g. repeated choices); phrase it as an inference about the user and cite every supporting fact. "
    "kind=summary aggregates related facts (totals, counts, timelines) across sessions without losing numbers. "
    "Never invent evidence or treat expired or negated facts as current."
)


async def prepare_materials(
    *, factory: ComponentFactory, app_id: str, user_id: str, agent_id: str
) -> dict:
    where = build_filter(
        app_ids=[app_id],
        user_id=user_id,
        agent_ids=[agent_id],
        layers=[Layer.L1_FACT, Layer.L2_PROFILE],
        statuses=[MemoryStatus.ACTIVE],
        latest_only=True,
    )
    facts = factory.vector.get_many(where, limit=100_001)
    if len(facts) > 100_000:
        raise DecisionContextError(
            "System2 scope exceeds scan budget; sources remain pending"
        )
    fresh = [f for f in facts if not f.s2_evidence_count]
    return {"facts": fresh, "scope": (app_id, user_id, agent_id)}


def _fact_payload(f: MemoryNode) -> dict:
    return {
        "id": f.node_id,
        "content": f.content,
        "memory_at": f.memory_at,
        "kind": f.kind,
        "valid_from": f.valid_from,
        "valid_until": f.valid_until,
        "update_type": (f.custom or {}).get("update_type"),
        "conflicts_with": (f.custom or {}).get("conflicts_with", []),
    }


def _build_user_message(materials: dict, *, max_facts: int = 0) -> str:
    facts = materials.get("facts", [])
    if max_facts:
        facts = facts[:max_facts]
    return json.dumps(
        {"facts": [_fact_payload(f) for f in facts]}, ensure_ascii=False
    )


def topic_order(facts: list[MemoryNode]) -> list[MemoryNode]:
    """Greedy nearest-neighbour chain over embeddings so topics stay adjacent.

    Pure ordering: every fact is kept. Facts without embeddings keep their place
    at the end in time order.
    """
    with_vec = [f for f in facts if f.embedding]
    rest = sorted(
        (f for f in facts if not f.embedding), key=lambda n: n.memory_at or n.gmt_created
    )
    if len(with_vec) <= 2:
        return with_vec + rest
    matrix = np.asarray([f.embedding for f in with_vec], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)
    similarity = matrix @ matrix.T
    start = min(
        range(len(with_vec)),
        key=lambda i: with_vec[i].memory_at or with_vec[i].gmt_created,
    )
    order = [start]
    remaining = set(range(len(with_vec))) - {start}
    while remaining:
        last = order[-1]
        nxt = max(remaining, key=lambda j: similarity[last, j])
        order.append(nxt)
        remaining.discard(nxt)
    return [with_vec[i] for i in order] + rest


class System2Agent:
    def __init__(self, *, factory: ComponentFactory) -> None:
        self.factory = factory

    def _batches(self, facts: list[MemoryNode]) -> list[list[MemoryNode]]:
        settings = self.factory.settings
        cap = settings.system2_max_prompt_facts
        char_budget = min(
            settings.llm_input_max_chars - 2500,
            (settings.jev_max_question_chars - 6000) // 3,
        )
        if char_budget <= 0:
            raise ValueError("System2 context budget is too small")
        batches: list[list[MemoryNode]] = []
        batch: list[MemoryNode] = []
        size = 0
        for fact in facts:
            item_size = len(json.dumps(_fact_payload(fact), ensure_ascii=False))
            if item_size > char_budget:
                raise ValueError(
                    "System2 fact exceeds generation context; retained pending"
                )
            if batch and (len(batch) >= cap or size + item_size > char_budget):
                batches.append(batch)
                batch, size = [], 0
            batch.append(fact)
            size += item_size
        if batch:
            batches.append(batch)
        return batches

    async def run(self, *, app_id: str, user_id: str, agent_id: str = "") -> dict:
        scope = (app_id, user_id, agent_id)
        scope_key = json.dumps(scope)
        await self._recover_projections(scope)
        material = await prepare_materials(
            factory=self.factory, app_id=app_id, user_id=user_id, agent_id=agent_id
        )
        settings = self.factory.settings
        quarantined = self.factory.cache.quarantined_s2_facts(
            scope=scope_key, max_attempts=settings.s2_max_attempts
        )
        fresh = [f for f in material["facts"] if f.node_id not in quarantined]
        with_vectors = self.factory.vector.get_by_ids([f.node_id for f in fresh])
        facts = topic_order([with_vectors.get(f.node_id, f) for f in fresh])
        stats = {
            "created_schemas": 0,
            "created_intentions": 0,
            "created_preferences": 0,
            "evidence_added": 0,
            "edges_added": 0,
            "failed_batches": 0,
            "quarantined_facts": len(quarantined),
            "retry_pending": 0,
        }
        # Batches within one scope stay ordered; disjoint scopes run concurrently in writer.
        pending = self._batches(facts)
        while pending:
            batch = pending.pop(0)
            ids = [f.node_id for f in batch]
            try:
                nodes = await self._process(batch, scope)
            except StaleDecisionError:
                raise
            except DecisionContextError as exc:
                stats["failed_batches"] += 1
                logger.warning(
                    "System2 scope=%s batch of %d failed: %s", scope, len(batch), exc
                )
                if len(batch) > 1:
                    half = len(batch) // 2
                    pending[:0] = [batch[:half], batch[half:]]
                else:
                    self.factory.cache.record_s2_failure(
                        scope=scope_key, fact_ids=ids, error=str(exc)
                    )
                    stats["retry_pending"] += 1
                continue
            self.factory.cache.clear_s2_failures(scope=scope_key, fact_ids=ids)
            stats["created_schemas"] += sum(
                n.kind not in ("intention", "preference") for n in nodes
            )
            stats["created_intentions"] += sum(n.kind == "intention" for n in nodes)
            stats["created_preferences"] += sum(n.kind == "preference" for n in nodes)
            stats["evidence_added"] += sum(
                len((n.custom or {}).get("source_ids", [])) for n in nodes
            )
        return stats

    async def _process(
        self, batch: list[MemoryNode], scope: tuple[str, str, str]
    ) -> list[MemoryNode]:
        app_id, user_id, agent_id = scope
        input_ids = {f.node_id for f in batch}
        stable_key = "s2:" + ":".join(sorted(input_ids))
        recovered = self.factory.vector.completed_request(stable_key)
        if recovered:
            nodes = list(
                self.factory.vector.get_by_ids(recovered["extra_node_ids"]).values()
            )
            await self._project(nodes)
            return nodes
        user = _build_user_message({"facts": batch})
        response = await self.factory.llm.chat_json(
            system=_SYSTEM, user=user, max_tokens=4096
        )
        if not isinstance(response, dict):
            raise DecisionContextError("invalid System2 generation response")
        processed = response.get("processed_ids")
        if not isinstance(processed, list) or set(processed) != input_ids:
            raise DecisionContextError("System2 did not explicitly cover its input IDs")
        drafts = response.get("memories")
        if not isinstance(drafts, list):
            raise DecisionContextError("System2 memories must be a list")
        for draft in drafts:
            if (
                not isinstance(draft, dict)
                or not isinstance(draft.get("content"), str)
                or not draft["content"].strip()
                or draft.get("kind", "schema") not in _KINDS
            ):
                raise DecisionContextError("invalid derived memory")
            evidence = draft.get("evidence_ids")
            if not isinstance(evidence, list) or not evidence or not set(evidence) <= input_ids:
                raise DecisionContextError("derived memory has missing or foreign evidence")
        questions = {
            f"draft:{i}": {
                "type": "choice",
                "instructions": f"Does derived draft index={i} follow from its cited evidence without inventing facts or treating inference as direct user speech? A preference draft must be supported by the pattern in its cited facts.",
                "criteria": {
                    "VALID": "Faithful and supported",
                    "INVALID": "Unsupported or misleading",
                },
            }
            for i in range(len(drafts))
        }
        questions["coverage"] = {
            "type": "choice",
            "instructions": "Are the generated derived memories faithful, and is it reasonable that facts with no derived memories have no useful cross-evidence inference? Check the actual sources, not just the processed_ids declaration.",
            "criteria": {
                "VALID": "Faithful processing",
                "INVALID": "Missing useful evidence or unsupported inference",
            },
        }
        checked = await self.factory.jev.evaluate(
            state={"facts": json.loads(user)["facts"], "generation": response},
            questions=questions,
        )
        if any(a.choice != "VALID" for a in checked.values()):
            raise DecisionContextError(
                "Jev rejected cognition or coverage; sources remain pending: "
                + ", ".join(
                    f"{key}={a.choice}" for key, a in checked.items() if a.choice != "VALID"
                )
            )
        lock = self.factory.commit_locks.get("decision::" + "::".join(scope))
        agent = MemAgent(factory=self.factory)
        meta = [
            {
                "content": d["content"],
                "layer": Layer.L3_DERIVED.value,
                "kind": d.get("kind", "schema"),
                "source_ids": d["evidence_ids"],
            }
            for d in drafts
        ]
        epoch = None
        if meta:
            ops = await agent.reconciler.reconcile(
                new_memories=[m["content"] for m in meta],
                new_memories_meta=meta,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                current_time=datetime.now(timezone.utc).isoformat(),
                source_content=user,
            )
            nodes, expected, archives = await agent._prepare_nodes(
                ops, batch[0], stable_key
            )
            epoch = ops[0].scope_epoch
        else:
            nodes, expected, archives = [], {}, []
        async with lock:
            current = self.factory.vector.get_by_ids(list(input_ids))
            if set(current) != input_ids or any(
                not current[f.node_id].is_latest
                or current[f.node_id].revision != f.revision
                for f in batch
            ):
                raise StaleDecisionError("System2 evidence changed during cognition")
            expected.update({f.node_id: f.revision for f in batch})
            if not self.factory.vector.apply_version_plan(
                nodes=nodes,
                expected=expected,
                archives=archives,
                scope=scope,
                expected_epoch=epoch,
                request_key=stable_key,
                processed_evidence=sorted(input_ids),
            ):
                raise StaleDecisionError("System2 targets changed; retry digest")
            await self._recover_projections(scope)
        return nodes

    async def _recover_projections(self, scope: tuple[str, str, str]) -> None:
        if self.factory.graph is None:
            return
        for plan_id, node_ids in self.factory.vector.pending_projections(scope):
            nodes = self.factory.vector.get_by_ids(node_ids)
            if set(nodes) != set(node_ids):
                raise DecisionContextError("committed projection is missing a version")
            await self._project(nodes.values())
            self.factory.vector.acknowledge_projection(plan_id)

    async def _project(self, nodes: Iterable[MemoryNode]) -> None:
        """Graph is a rebuildable projection of committed VDB versions/evidence."""
        graph = self.factory.graph
        if graph is None:
            return
        for node in nodes:
            graph.add_node(
                GraphNode(
                    node_id=node.node_id,
                    layer=node.layer.value,
                    content=node.content,
                    app_id=node.app_id,
                    user_id=node.user_id,
                    agent_id=node.agent_id,
                    embedding=node.embedding or [],
                    tags=node.tags,
                    custom={**(node.custom or {}), "kind": node.kind},
                    gmt_created=node.gmt_created,
                )
            )
            for fid in (node.custom or {}).get("source_ids", []):
                graph.add_evidence(schema_id=node.node_id, fact_id=fid)
