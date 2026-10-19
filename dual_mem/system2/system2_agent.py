"""Cross-evidence cognition: one generation, one validation, append.

Dual mode adds L3 derived memories on top of the System1 write path. This module keeps
that as cheap as possible:

- facts are grouped by topic (embedding nearest-neighbour) and packed into as few
  batches as the generation model's context allows;
- each batch is one LLM generation + one Jev validation of the drafts it produced;
- drafts are appended as standalone L3 nodes. They do **not** go through relation
  decisions: derived memories never supersede facts, so choosing a target among all
  current heads is pure cost. A draft that duplicates an existing derived memory is
  skipped by embedding similarity;
- **there is no retry loop**. A batch that fails (bad JSON, foreign evidence, Jev
  rejection) is logged and dropped; its facts stay eligible for a later digest, but
  nothing re-runs inside the same digest and nothing accumulates in a failure table.
  Stability of the write path matters more than covering every inference.

Facts are marked processed (``s2_evidence_count``) only when their batch commits, so a
dropped batch is naturally retried by the *next* digest without any bookkeeping.
"""

from __future__ import annotations

import json
import asyncio
import logging
import uuid
from datetime import datetime, timezone

import numpy as np

from dual_mem.errors import DecisionContextError
from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.system2.agent")

_KINDS = ("schema", "summary", "intention", "preference")
_SYSTEM = (
    "Generate cross-evidence memories, not relation decisions. Use only supplied fact IDs. "
    'Return {"memories":[{"content":...,"kind":"schema|summary|intention|preference","evidence_ids":[...]}],'
    '"processed_ids":[...]}. List every input ID you examined. '
    "kind=preference is a user preference, taste, habit or value that the cited facts imply but do not state "
    "outright (e.g. repeated choices); phrase it as an inference about the user and cite every supporting fact. "
    "kind=summary aggregates related facts (totals, counts, timelines) across sessions without losing numbers. "
    "Capture WHY choices changed, the constraints behind them, and contrasts between earlier and current values, "
    "not just generic personality labels. Prefer a few useful, specific cross-evidence memories over restating each fact. "
    "For preferences, learn the evidence-backed decision principle: what feature or constraint drives the choice, "
    "in which situation it applies, and what changes or exceptions limit it. Write a self-contained conditional "
    "preference when supported, rather than only listing activities the user did. "
    "Keep domain and situation scope explicit. The same user having a preference in one context does not prove "
    "it transfers to an unrelated context; a context-independent preference needs corroboration across contexts. "
    "Do not invent an avoidance, recommendation or condition just to fill a template. "
    "Keep dates and distinguish completed events from plans. Assistant advice is not evidence that the user has a "
    "preference, possession or experience. Never invent evidence or treat expired or negated facts as current."
    " A fact may contain history snapshots: these are SUPERSEDED, not current statements. "
    "Use them to explain evolution and reasons, citing the enclosing fact ID. Never promote an old value back to current. "
    "Use as_of for validity checks; it is the conversation evidence cutoff, not generation time."
)


def _fact_payload(f: MemoryNode, fid: str | None = None) -> dict:
    return {
        "id": f.node_id if fid is None else fid,
        "content": f.content,
        "memory_at": f.memory_at,
        "conversation_date": datetime.fromtimestamp(f.memory_at, tz=timezone.utc).isoformat() if f.memory_at is not None else None,
        "owner": f.owner,
        "kind": f.kind,
        "valid_from": f.valid_from,
        "valid_until": f.valid_until,
        "update_type": (f.custom or {}).get("update_type"),
        "conflicts_with": (f.custom or {}).get("conflicts_with", []),
    }


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


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / denom) if denom else 0.0


class System2Agent:
    def __init__(self, *, factory: ComponentFactory) -> None:
        self.factory = factory
        self._histories: dict[str, list[MemoryNode]] = {}

    # ---- material -------------------------------------------------------------

    def prepare(self, *, app_id: str, user_id: str, agent_id: str) -> list[MemoryNode]:
        """Facts not yet used by cognition, newest last, ordered by topic."""
        where = build_filter(
            app_ids=[app_id],
            user_id=user_id,
            agent_ids=[agent_id],
            layers=[Layer.L1_FACT, Layer.L2_PROFILE],
            statuses=[MemoryStatus.ACTIVE],
            latest_only=True,
        )
        facts = self.factory.vector.get_many(where, limit=100_001)
        fresh = [f for f in facts if not f.s2_evidence_count]
        if not fresh:
            return []
        with_vectors = self.factory.vector.get_by_ids([f.node_id for f in fresh])
        fresh = [with_vectors.get(f.node_id, f) for f in fresh]
        self._load_histories(fresh, (app_id, user_id, agent_id))
        return topic_order(fresh)

    def _load_histories(self, facts: list[MemoryNode], scope: tuple[str, str, str]) -> None:
        known = {f.node_id: f for f in facts}
        pending = {nid for f in facts for nid in f.supersedes} - known.keys()
        while pending:
            nodes = self.factory.vector.get_by_ids(sorted(pending))
            if set(nodes) != pending:
                raise DecisionContextError("missing System2 history evidence")
            if any((n.app_id, n.user_id, n.agent_id) != scope for n in nodes.values()):
                raise DecisionContextError("System2 history crosses tenant scope")
            known.update(nodes)
            pending = {nid for n in nodes.values() for nid in n.supersedes} - known.keys()
        self._histories = {}
        for fact in facts:
            seen: set[str] = set()
            pending = list(fact.supersedes)
            while pending:
                nid = pending.pop()
                if nid == fact.node_id:
                    raise DecisionContextError("cyclic System2 history")
                if nid in seen:
                    continue
                seen.add(nid)
                pending.extend(known[nid].supersedes)
            self._histories[fact.node_id] = sorted(
                (known[nid] for nid in seen if known[nid].layer in (Layer.L1_FACT, Layer.L2_PROFILE)),
                key=lambda n: n.memory_at or n.gmt_created, reverse=True,
            )
            if any(n.is_latest for n in self._histories[fact.node_id]):
                raise DecisionContextError("System2 history contains a current version")

    def _material_payload(self, fact: MemoryNode, fid: str) -> dict:
        payload = _fact_payload(fact, fid)
        history = self._histories.get(fact.node_id, [])
        if history:
            payload["history"] = [
                {k: v for k, v in _fact_payload(n).items() if k != "id"} | {"is_latest": False}
                for n in history
            ]
        return payload

    def _batches(self, facts: list[MemoryNode]) -> list[list[MemoryNode]]:
        settings = self.factory.settings
        cap = settings.system2_max_prompt_facts
        char_budget = min(
            settings.llm_input_max_chars - 2500,
            settings.jev_max_question_chars - 6000,
        )
        if char_budget <= 0:
            raise ValueError("System2 context budget is too small")
        batches: list[list[MemoryNode]] = []
        batch: list[MemoryNode] = []
        size = 0
        for fact in facts:
            item_size = len(json.dumps(self._material_payload(fact, f"f{len(batch)}"), ensure_ascii=False))
            if item_size > char_budget:
                # One oversized fact must not stall the whole scope.
                logger.warning("System2 fact %s exceeds budget; skipped", fact.node_id)
                continue
            if batch and (len(batch) >= cap or size + item_size > char_budget):
                batches.append(batch)
                batch, size = [], 0
            batch.append(fact)
            size += item_size
        if batch:
            batches.append(batch)
        return batches

    # ---- run ------------------------------------------------------------------

    async def run(self, *, app_id: str, user_id: str, agent_id: str = "") -> dict:
        stats = {
            "batches": 0,
            "failed_batches": 0,
            "created_schemas": 0,
            "created_intentions": 0,
            "created_preferences": 0,
            "duplicates_skipped": 0,
        }
        await self._recover_projections((app_id, user_id, agent_id))
        facts = self.prepare(app_id=app_id, user_id=user_id, agent_id=agent_id)
        if not facts:
            return stats
        existing = self._existing_derived(app_id=app_id, user_id=user_id, agent_id=agent_id)
        for batch in self._batches(facts):
            stats["batches"] += 1
            try:
                created, skipped = await self._process(
                    batch, (app_id, user_id, agent_id), existing
                )
            except (DecisionContextError, ValueError) as exc:
                # Dropped, not retried: the facts keep s2_evidence_count == 0 and are
                # picked up by the next digest.
                stats["failed_batches"] += 1
                logger.warning(
                    "System2 scope=(%s,%s,%s) batch of %d dropped: %s",
                    app_id,
                    user_id,
                    agent_id,
                    len(batch),
                    exc,
                )
                continue
            stats["duplicates_skipped"] += skipped
            stats["created_schemas"] += sum(n.kind not in ("intention", "preference") for n in created)
            stats["created_intentions"] += sum(n.kind == "intention" for n in created)
            stats["created_preferences"] += sum(n.kind == "preference" for n in created)
        return stats

    def _existing_derived(
        self, *, app_id: str, user_id: str, agent_id: str
    ) -> list[MemoryNode]:
        where = build_filter(
            app_ids=[app_id],
            user_id=user_id,
            agent_ids=[agent_id],
            layers=[Layer.L3_DERIVED],
            statuses=[MemoryStatus.ACTIVE],
            latest_only=True,
        )
        nodes = self.factory.vector.get_many(where, limit=5000)
        embedded = self.factory.vector.get_by_ids([n.node_id for n in nodes])
        return [embedded.get(n.node_id, n) for n in nodes]

    async def _process(
        self,
        batch: list[MemoryNode],
        scope: tuple[str, str, str],
        existing: list[MemoryNode],
    ) -> tuple[list[MemoryNode], int]:
        app_id, user_id, agent_id = scope
        by_id = {f"f{i}": f for i, f in enumerate(batch)}
        facts = {fid: self._material_payload(fact, fid) for fid, fact in by_id.items()}
        input_ids = set(by_id)
        cutoff = max((f.memory_at for f in batch if f.memory_at is not None), default=None)
        as_of = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat() if cutoff is not None else None
        user = json.dumps({"facts": list(facts.values()), "as_of": as_of}, ensure_ascii=False)
        response = await self.factory.llm.chat_json(system=_SYSTEM, user=user, max_tokens=4096, temperature=0.0)
        if not isinstance(response, dict):
            raise DecisionContextError("invalid System2 generation response")
        drafts = response.get("memories")
        if not isinstance(drafts, list):
            raise DecisionContextError("System2 memories must be a list")

        clean: list[dict] = []
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
            clean.append(draft)

        if clean:
            async def validate(i: int, draft: dict) -> bool:
                verdicts = await self.factory.jev.evaluate(
                    state={"facts": [facts[fid] for fid in draft["evidence_ids"]], "draft": draft, "as_of": as_of},
                    questions={
                    f"draft:{i}": {
                        "type": "choice",
                        "instructions": f"Does derived draft index={i} follow from its cited evidence without inventing facts or treating inference as direct user speech? A preference draft must be supported by the pattern in its cited facts and respect current values, conditions and domain scope. Reject unsupported transfer to other contexts or invented decision rules. History snapshots may support an evolution or causal summary, but are superseded, not current values. Check validity at as_of, not generation time.",
                        "criteria": {
                            "VALID": "Faithful and supported",
                            "INVALID": "Unsupported or misleading",
                        },
                    }
                    },
                )
                return verdicts[f"draft:{i}"].choice == "VALID"
            # Unsupported drafts are dropped; supported ones are kept regardless of
            # what Jev thinks about the batch as a whole.
            accepted = await asyncio.gather(*(validate(i, draft) for i, draft in enumerate(clean)))
            clean = [draft for draft, valid in zip(clean, accepted, strict=True) if valid]
        if not clean:
            await self._acknowledge(batch, scope)
            return [], 0

        embeddings = await self.factory.embed.embed_batch([d["content"] for d in clean])
        nodes: list[MemoryNode] = []
        skipped = 0
        for draft, vector in zip(clean, embeddings):
            if any(_cosine(vector, e.embedding) >= self.factory.settings.s2_dedup_threshold
                   for e in [*existing, *nodes] if e.embedding):
                skipped += 1
                continue
            evidence = sorted({
                node.node_id: node
                for fid in draft["evidence_ids"]
                for node in [by_id[fid], *self._histories.get(by_id[fid].node_id, [])]
            }.values(), key=lambda n: n.node_id)
            evidence_ids = [n.node_id for n in evidence]
            nodes.append(
                MemoryNode(
                    node_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"s2:{app_id}:{user_id}:{agent_id}:{draft.get('kind', 'schema')}:{draft['content']}:{':'.join(evidence_ids)}")),
                    content=draft["content"],
                    layer=Layer.L3_DERIVED,
                    kind=draft.get("kind", "schema"),
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    memory_at=max((n.memory_at for n in evidence if n.memory_at is not None), default=None),
                    custom={
                        "source_ids": evidence_ids,
                        "source_node_id": min(by_id[fid].node_id for fid in draft["evidence_ids"]),
                    },
                    embedding=vector,
                )
            )
        await self._commit(nodes, batch, scope)
        existing.extend(nodes)
        return nodes, skipped

    async def _acknowledge(
        self, batch: list[MemoryNode], scope: tuple[str, str, str]
    ) -> None:
        """Mark facts processed when a batch legitimately yields no derived memory."""
        await self._commit([], batch, scope)

    async def _commit(
        self,
        nodes: list[MemoryNode],
        batch: list[MemoryNode],
        scope: tuple[str, str, str],
    ) -> None:
        ok = await self._apply(nodes, batch, scope)
        if not ok:
            raise DecisionContextError("System2 commit raced with a concurrent write")
        await self._recover_projections(scope)

    async def _recover_projections(self, scope: tuple[str, str, str]) -> None:
        """Project every committed L3 plan of ``scope`` not yet in the graph.

        The projection intent is journaled with the version plan, so a graph failure
        (or crash) after commit is repaired here even though the facts are already
        marked processed and will not be regenerated.
        """
        if self.factory.graph is None:
            return
        for plan_id, node_ids in self.factory.vector.pending_projections(scope):
            nodes = self.factory.vector.get_by_ids(node_ids)
            if set(nodes) != set(node_ids):
                raise DecisionContextError("committed projection is missing a version")
            await self._project(list(nodes.values()))
            self.factory.vector.acknowledge_projection(plan_id)

    async def _apply(
        self,
        nodes: list[MemoryNode],
        batch: list[MemoryNode],
        scope: tuple[str, str, str],
    ) -> bool:
        async with self.factory.commit_locks.get("decision::" + "::".join(scope)):
            return await asyncio.to_thread(
                self.factory.vector.apply_version_plan,
                nodes=nodes,
                # Optimistic CAS on exactly the facts this batch reasoned over: if one
                # was superseded meanwhile, the plan is rejected rather than deriving
                # from stale evidence.
                expected={f.node_id: f.revision for f in batch},
                archives=[],
                scope=scope,
                request_key="s2:" + ":".join(sorted(f.node_id for f in batch)),
                processed_evidence=sorted(f.node_id for f in batch),
            )

    async def _project(self, nodes: list[MemoryNode]) -> None:
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


__all__ = ["System2Agent", "topic_order"]
