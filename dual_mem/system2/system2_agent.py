"""System 2: offline cognition over System 1 memories (aligned with hy-memory 2.0.4).

Three question-blind products, all written by ``digest``:

1. **Basic schemas** — "When <circumstance>, the user <pattern> — reflecting <insight>."
   One atomic, domain-bound behavioural pattern, induced from one embedding cluster of
   *fresh* current facts. Content is immutable; later support is ``add_evidence``.
2. **Core schemas** — cross-domain patterns. Each basic schema gets a behaviour-only
   abstraction (domain nouns stripped) embedded into a second space; pairs that are
   similar in behaviour but dissimilar in content "collide", collisions are grouped
   by union-find, and one higher-order pattern is induced per group (or none).
3. **Persona catalog** — exact current facts selected (never rewritten) into a fixed
   domain taxonomy, a few per domain, so a reader can see a balanced profile of who
   the user currently is regardless of what the query happens to retrieve.

Write path for basic schemas, per scope:

* **No-LLM preparation.** Current (ACTIVE, latest) facts/profiles not yet reviewed by an
  earlier digest are clustered: average-linkage cosine, oversize clusters re-split at a
  stricter distance, near-duplicates collapsed, clusters smaller than
  ``system2_min_cluster_size`` dropped. No cluster ⇒ no LLM call.
* **One LLM call per cluster** returning operations over the cluster's ids:
  ``create_schema``, ``add_evidence`` (to an existing schema it was shown), or none.
* **Structural gates.** Cited ids must be in the cluster and user-owned, and must span
  ``system2_min_sources`` distinct raw sources. A near-duplicate of an existing schema
  that also shares evidence becomes ``add_evidence``. Nothing is repaired or retried.

Superseded versions never feed cognition: a schema is a pattern of the user's current
memory. All judgements are made by models or structure, never by lexical rules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import numpy as np

from dual_mem.errors import DecisionContextError
from dual_mem.coding.types import PROCEDURE_KIND
from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.system2.agent")

SCHEMA_KIND = "schema"
CORE_SCHEMA_KIND = "core_schema"
CATALOG_KIND = "persona_catalog"
SCHEMA_KINDS = (SCHEMA_KIND, CORE_SCHEMA_KIND)

CATALOG_DOMAINS = {
    "identity_relationships": "identity and the user's own family/relationship structure",
    "health_accessibility": "explicit health, sensory, mobility, ergonomic constraints",
    "preferences_collections": "explicit durable likes, dislikes, tastes, collections",
    "habits_items": "recurring habits and personally owned/carried/used items",
    "work_education": "durable role, expertise, education, recurring responsibilities",
    "culture_creative_experience": "concrete cultural/creative interests and strong experiences",
    "values_coping": "durable values and personally practiced coping resources",
    "current_constraints_goals": "ongoing burdens, constraints, or durable active goals",
}

_SCHEMA_SYSTEM = """You are a cognitive processing agent. You evolve higher-order cognitive structures from a user's memories.

A Schema captures ONE behavioral pattern in a SPECIFIC domain. Three components:
- Circumstance: the domain/topic/situation where this pattern is observed. Stay within it; do NOT generalize across domains (cross-domain abstraction is handled by a separate process).
- Pattern: the user's habitual behavior, thinking style, or action tendency in this circumstance.
- Insight: the underlying psychological driver or mental model.

Content format, a single sentence: "When [circumstance], the user [pattern] — reflecting [insight]."

Rules:
- Atomic: one pattern per Schema. Two distinct patterns are two Schemas.
- Immutable: an existing Schema is never rewritten. If the cluster supports an existing Schema, use add_evidence; do NOT recreate it.
- Evidence: a Schema requires user-owned evidence from at least two independent raw sources (each memory lists its "source"). Create a user Schema only when the material explicitly attributes the pattern to the user; assistant suggestions are not evidence about the user.
- Prefer not creating over creating low-quality nodes. A restated single fact, or a vague trait without circumstance, is low quality.

Good: "When cooking, the user strictly follows recipe steps and precisely measures ingredients — reflecting a need for external structure to manage uncertainty."
Bad: "The user approaches music as a medium for creative synthesis and cultural bridge-building, consistently blending diverse musical traditions..." (multiple patterns merged, too broad)
Bad: "The user is passionate about many things and values quality." (no circumstance, too vague)

Return strict JSON:
{"operations": [
  {"op": "create_schema", "content": "<one sentence>", "evidence_ids": ["m0", "m3"]},
  {"op": "add_evidence", "schema_id": "s0", "evidence_ids": ["m2"]}
]}
An empty list is a valid answer."""

_BEHAVIOR_SYSTEM = """You are a behavioral psychologist. Abstract the schema into a pure psychological/behavioral description. Strip ALL domain-specific nouns and scenarios; output ONLY the underlying behavioral style and psychological motivation, in the same language as the input.
Return strict JSON: {"abstraction": "<pure behavioral description>"}"""

_CORE_SYSTEM = """You are a deep pattern analyst. The following schemas come from different areas of the user's life. On the surface they appear unrelated, but their underlying behavioral logic is strikingly similar.
Synthesize ONE higher-order pattern that explains why these behaviors co-occur: a deep cognitive style, core psychological need, or hidden mental model. Only output a synthesis if the connection is genuinely compelling and logically airtight; if the evidence is weak or the connection superficial, return null. Use the same language as the input.
Return strict JSON: {"core_pattern": "<one sentence>" | null}"""

_CATALOG_SYSTEM = """Build a balanced long-term persona catalog from numbered recorded facts about the user.
Return strict JSON only: {"domains": {"<domain>": [<entry numbers>], ...}}

Use exactly these domain keys:
""" + "\n".join(f"- {k}: {v}" for k, v in CATALOG_DOMAINS.items()) + """

Select at most {per_domain} entry numbers per domain and at most {total} unique entries in total. Prefer specific, discriminative, user-owned facts that could materially personalize future answers. Preserve diversity within and across domains. Reject assistant advice, generic questions or curiosity without a personal fact, temporary logistics, one-off requests, and facts mainly about another person. Do not infer, summarize, rewrite, or create facts. An entry may appear in at most one domain. Output entry numbers only."""


def raw_source(node: MemoryNode) -> str:
    """The raw episode a memory was extracted from (its independent-evidence unit)."""
    return str((node.custom or {}).get("source_node_id") or node.session_id or node.node_id)


def _unit(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norm == 0, 1.0, norm)


def _average_linkage(matrix: np.ndarray, *, distance: float) -> list[list[int]]:
    from sklearn.cluster import AgglomerativeClustering

    if len(matrix) < 2:
        return [list(range(len(matrix)))]
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=distance,
    ).fit_predict(matrix)
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)
    return list(groups.values())


def cluster_facts(
    nodes: list[MemoryNode], *, distance: float, refine_distance: float, min_size: int,
    max_size: int, dedup_cosine: float,
) -> list[list[MemoryNode]]:
    """Average-linkage cosine clusters; oversized ones re-split at a stricter distance.

    Density clustering (DBSCAN) chains one user's memories into a single component,
    because every memory about the same person is moderately similar to every other.
    Average linkage keeps groups topically coherent. Near-duplicates are collapsed and
    clusters smaller than ``min_size`` are not induced.
    """
    vec = [n for n in nodes if n.embedding]
    if len(vec) < min_size:
        return []
    matrix = _unit(np.asarray([n.embedding for n in vec], dtype=float))
    clusters: list[list[int]] = []
    for members in _average_linkage(matrix, distance=distance):
        if len(members) <= max_size:
            clusters.append(members)
            continue
        for sub in _average_linkage(matrix[members], distance=refine_distance):
            ordered = [members[i] for i in sub]
            # Still oversized at the stricter distance: keep the members closest to
            # the sub-cluster centre (one bounded prompt), the rest stay fresh.
            clusters.append(ordered)
    out: list[list[MemoryNode]] = []
    for members in clusters:
        kept: list[int] = []
        for i in members:
            if all(float(matrix[i] @ matrix[j]) < dedup_cosine for j in kept):
                kept.append(i)
        if len(kept) < min_size:
            continue
        centroid = matrix[kept].mean(axis=0)
        kept.sort(key=lambda i: -float(matrix[i] @ centroid))
        out.append([vec[i] for i in kept[:max_size]])
    return out


def collision_groups(
    content: np.ndarray, behaviour: np.ndarray, *, beh_threshold: float, sem_threshold: float,
) -> list[list[int]]:
    """Union-find over pairs similar in behaviour but dissimilar in content."""
    n = len(content)
    if n < 2:
        return []
    c, b = _unit(content), _unit(behaviour)
    mask = ((b @ b.T) > beh_threshold) & ((c @ c.T) < sem_threshold) & ~np.eye(n, dtype=bool)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in np.argwhere(mask):
        if i < j:
            parent[find(int(i))] = find(int(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(va @ vb / denom) if denom else 0.0


def _derived_id(scope: tuple[str, str, str], kind: str, content: str, evidence: list[str]) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"s2:{kind}:{':'.join(scope)}:{content}:{':'.join(evidence)}"))


class System2Agent:
    def __init__(self, *, factory: ComponentFactory) -> None:
        self.factory = factory

    # ---- material -------------------------------------------------------------

    def _where(self, scope: tuple[str, str, str], layers: list[Layer]) -> dict:
        return build_filter(
            app_ids=[scope[0]], user_id=scope[1], agent_ids=[scope[2]],
            layers=layers, statuses=[MemoryStatus.ACTIVE], latest_only=True,
        )

    def current_facts(self, scope: tuple[str, str, str]) -> list[MemoryNode]:
        """Current System 1 memories with embeddings."""
        nodes = self.factory.vector.get_many(self._where(scope, [Layer.L1_FACT, Layer.L2_PROFILE]), limit=100_001)
        embedded = self.factory.vector.get_by_ids([n.node_id for n in nodes])
        return [embedded.get(n.node_id, n) for n in nodes]

    def derived(self, scope: tuple[str, str, str], kinds: tuple[str, ...]) -> list[MemoryNode]:
        nodes = [n for n in self.factory.vector.get_many(self._where(scope, [Layer.L3_DERIVED]), limit=5000)
                 if n.kind in kinds]
        embedded = self.factory.vector.get_by_ids([n.node_id for n in nodes])
        return [embedded.get(n.node_id, n) for n in nodes]

    # ---- run ------------------------------------------------------------------

    async def run(self, *, app_id: str, user_id: str, agent_id: str = "") -> dict:
        scope = (app_id, user_id, agent_id)
        stats = {
            "candidate_facts": 0, "clusters": 0, "failed_clusters": 0, "created_schemas": 0,
            "evidence_added": 0, "rejected_drafts": 0, "core_groups": 0, "created_core": 0,
            "catalog_entries": 0,
        }
        await self._recover_projections(scope)
        facts = self.current_facts(scope)
        own = [f for f in facts if f.owner != "assistant"]
        fresh = [f for f in own if not f.s2_evidence_count]
        stats["candidate_facts"] = len(fresh)
        settings = self.factory.settings
        clusters = cluster_facts(
            fresh, distance=settings.system2_cluster_distance,
            refine_distance=settings.system2_refine_distance,
            min_size=settings.system2_min_cluster_size, max_size=settings.system2_max_prompt_facts,
            dedup_cosine=settings.system2_dedup_cosine,
        )
        stats["clusters"] = len(clusters)
        schemas = self.derived(scope, (SCHEMA_KIND,))
        # Clusters are processed sequentially so each sees the schemas the previous
        # ones created (add_evidence instead of near-duplicates), as in hy-memory.
        for cluster in clusters:
            try:
                result = await self._induce(cluster, scope, schemas)
            except (DecisionContextError, ValueError) as exc:
                logger.warning("System2 scope=%s cluster of %d dropped: %s", scope, len(cluster), exc)
                stats["failed_clusters"] += 1
                continue
            for key in ("created_schemas", "evidence_added", "rejected_drafts"):
                stats[key] += result[key]
        clustered = {f.node_id for c in clusters for f in c}
        reviewed = [f for f in fresh if f.node_id not in clustered]
        if reviewed:
            # Facts no cluster could use are reviewed; they are revisited only once they
            # are superseded (a new version is a new, unreviewed memory).
            await self._commit([], reviewed, scope)
        # The sweeper and the catalog are independent products of the same digest: a
        # failed one is dropped (and retried by the next digest), never fatal.
        if settings.system2_cross_domain_enabled:
            try:
                stats["core_groups"], stats["created_core"] = await self._cross_domain(scope)
            except (DecisionContextError, ValueError) as exc:
                logger.warning("System2 scope=%s cross-domain sweep dropped: %s", scope, exc)
        if settings.system2_catalog_enabled:
            try:
                # The persona catalog describes the person; engineering procedures
                # still feed schemas above but are not persona facts.
                stats["catalog_entries"] = await self._catalog(
                    scope, [f for f in own if f.kind != PROCEDURE_KIND]
                )
            except (DecisionContextError, ValueError) as exc:
                logger.warning("System2 scope=%s persona catalog dropped: %s", scope, exc)
        return stats

    # ---- basic schemas --------------------------------------------------------

    async def _induce(
        self, cluster: list[MemoryNode], scope: tuple[str, str, str], schemas: list[MemoryNode],
    ) -> dict:
        settings = self.factory.settings
        by_mid = {f"m{i}": f for i, f in enumerate(cluster)}
        sources = {mid: raw_source(f) for mid, f in by_mid.items()}
        labels = {src: f"src{i}" for i, src in enumerate(dict.fromkeys(sources.values()))}
        centroid = np.mean([f.embedding for f in cluster], axis=0).tolist()
        near = sorted((s for s in schemas if s.embedding), key=lambda s: -_cosine(centroid, s.embedding))
        by_sid = {f"s{i}": s for i, s in enumerate(near[: settings.system2_existing_per_group])}
        user = json.dumps({
            "cluster": [
                {"id": mid, "source": labels[sources[mid]], "content": f.content, "kind": f.kind}
                for mid, f in by_mid.items()
            ],
            "existing_schemas": [
                {"id": sid, "content": s.content, "evidence": len((s.custom or {}).get("source_ids", []))}
                for sid, s in by_sid.items()
            ],
        }, ensure_ascii=False)
        response = await self.factory.llm.chat_json(
            system=_SCHEMA_SYSTEM, user=user, max_tokens=settings.system2_max_tokens, temperature=0.0,
        )
        operations = response.get("operations") if isinstance(response, dict) else None
        if not isinstance(operations, list):
            raise DecisionContextError("System2 operations must be a list")

        rejected = 0
        additions: dict[str, set[str]] = {}
        drafts: list[tuple[str, list[str]]] = []
        for op in operations:
            evidence = op.get("evidence_ids") if isinstance(op, dict) else None
            if not isinstance(evidence, list) or not evidence or not set(evidence) <= by_mid.keys():
                rejected += 1
                continue
            if op.get("op") == "add_evidence" and op.get("schema_id") in by_sid:
                additions.setdefault(op["schema_id"], set()).update(evidence)
                continue
            content = op.get("content")
            if (op.get("op") != "create_schema" or not isinstance(content, str) or not content.strip()
                    or len({sources[m] for m in evidence}) < settings.system2_min_sources):
                rejected += 1
                continue
            drafts.append((content.strip(), sorted(set(evidence))))

        new_nodes: list[MemoryNode] = []
        if drafts:
            vectors = await self.factory.embed.embed_batch([c for c, _ in drafts])
            for (content, mids), vector in zip(drafts, vectors, strict=True):
                ids = {by_mid[m].node_id for m in mids}
                twin = max(
                    ((s, _cosine(vector, s.embedding)) for s in [*schemas, *new_nodes] if s.embedding),
                    key=lambda t: t[1], default=(None, 0.0),
                )
                if twin[0] is not None and twin[1] >= settings.system2_schema_dedup_cosine and (
                    ids & set((twin[0].custom or {}).get("source_ids", []))
                ):
                    sid = next((k for k, v in by_sid.items() if v.node_id == twin[0].node_id), None)
                    if sid is not None:
                        additions.setdefault(sid, set()).update(mids)
                    else:
                        rejected += 1
                    continue
                evidence = [by_mid[m] for m in mids]
                evidence_ids = sorted(f.node_id for f in evidence)
                new_nodes.append(MemoryNode(
                    node_id=_derived_id(scope, SCHEMA_KIND, content, evidence_ids),
                    content=content, layer=Layer.L3_DERIVED, kind=SCHEMA_KIND,
                    app_id=scope[0], user_id=scope[1], agent_id=scope[2],
                    memory_at=max((f.memory_at for f in evidence if f.memory_at is not None), default=None),
                    custom={
                        "source_ids": evidence_ids,
                        "raw_source_ids": sorted({raw_source(f) for f in evidence}),
                        "source_node_id": min(evidence_ids),
                    },
                    embedding=vector,
                ))
        edges = []
        for sid, mids in additions.items():
            schema = by_sid[sid]
            current = set((schema.custom or {}).get("source_ids", []))
            edges.extend((schema.node_id, by_mid[m].node_id) for m in sorted(mids)
                         if by_mid[m].node_id not in current)
        await self._commit(new_nodes, cluster, scope)
        if edges and self.factory.graph is not None:
            for schema_id, fact_id in edges:
                self.factory.graph.add_evidence(schema_id=schema_id, fact_id=fact_id)
        schemas.extend(new_nodes)
        return {"created_schemas": len(new_nodes), "evidence_added": len(edges), "rejected_drafts": rejected}

    # ---- core schemas ---------------------------------------------------------

    async def _cross_domain(self, scope: tuple[str, str, str]) -> tuple[int, int]:
        settings = self.factory.settings
        basics = [s for s in self.derived(scope, (SCHEMA_KIND,)) if s.embedding]
        if len(basics) < settings.system2_cross_domain_min_basics:
            return 0, 0
        missing = [s for s in basics if not (s.custom or {}).get("behavior_text")]

        async def abstract(node: MemoryNode) -> str | None:
            response = await self.factory.llm.chat_json(
                system=_BEHAVIOR_SYSTEM, user=node.content, max_tokens=256, temperature=0.0,
            )
            text = response.get("abstraction") if isinstance(response, dict) else None
            return text.strip() if isinstance(text, str) and text.strip() else None

        texts = await asyncio.gather(*(abstract(s) for s in missing))
        written = {s.node_id: t for s, t in zip(missing, texts, strict=True) if t}
        if written:
            # The abstraction is a derived index of the schema, not its content; it is
            # kept on the schema so later sweeps reuse it (the vector is re-embedded).
            patches = {}
            for node in missing:
                if node.node_id in written:
                    node.custom = {**(node.custom or {}), "behavior_text": written[node.node_id]}
                    patches[node.node_id] = {"custom": json.dumps(node.custom, ensure_ascii=False)}
            await asyncio.to_thread(self.factory.vector.update_payload_many, patches)
        ready = [s for s in basics if (s.custom or {}).get("behavior_text")]
        if len(ready) < 2:
            return 0, 0
        behaviour = await self.factory.embed.embed_batch([s.custom["behavior_text"] for s in ready])
        groups = collision_groups(
            np.asarray([s.embedding for s in ready], dtype=float),
            np.asarray(behaviour, dtype=float),
            beh_threshold=settings.system2_collision_beh_threshold,
            sem_threshold=settings.system2_collision_sem_threshold,
        )
        existing = {tuple(sorted((c.custom or {}).get("member_schema_ids", [])))
                    for c in self.derived(scope, (CORE_SCHEMA_KIND,))}
        created: list[MemoryNode] = []
        for group in groups:
            members = sorted((ready[i] for i in group), key=lambda s: s.node_id)
            member_ids = tuple(s.node_id for s in members)
            if member_ids in existing:
                continue
            response = await self.factory.llm.chat_json(
                system=_CORE_SYSTEM,
                user=json.dumps({"schemas": [s.content for s in members]}, ensure_ascii=False),
                max_tokens=512, temperature=0.0,
            )
            pattern = response.get("core_pattern") if isinstance(response, dict) else None
            if not isinstance(pattern, str) or not pattern.strip():
                continue
            evidence_ids = sorted({fid for s in members for fid in (s.custom or {}).get("source_ids", [])})
            [vector] = await self.factory.embed.embed_batch([pattern.strip()])
            created.append(MemoryNode(
                node_id=_derived_id(scope, CORE_SCHEMA_KIND, pattern.strip(), list(member_ids)),
                content=pattern.strip(), layer=Layer.L3_DERIVED, kind=CORE_SCHEMA_KIND,
                app_id=scope[0], user_id=scope[1], agent_id=scope[2],
                memory_at=max((s.memory_at for s in members if s.memory_at is not None), default=None),
                custom={
                    "source_ids": evidence_ids,
                    "member_schema_ids": list(member_ids),
                    "raw_source_ids": sorted({r for s in members for r in (s.custom or {}).get("raw_source_ids", [])}),
                    "source_node_id": member_ids[0],
                },
                embedding=vector,
            ))
        if created:
            await self._commit(created, [], scope)
        return len(groups), len(created)

    # ---- persona catalog ------------------------------------------------------

    async def _catalog(self, scope: tuple[str, str, str], facts: list[MemoryNode]) -> int:
        settings = self.factory.settings
        bank = sorted(facts, key=lambda f: f.node_id)[: settings.system2_catalog_max_facts]
        if len(bank) < settings.system2_catalog_min_facts:
            return 0
        fingerprint = str(uuid.uuid5(uuid.NAMESPACE_URL, ":".join(f"{f.node_id}@{f.revision}" for f in bank)))
        current = self.derived(scope, (CATALOG_KIND,))
        if any((c.custom or {}).get("fingerprint") == fingerprint for c in current):
            return 0
        lines = [f"{i}. {' '.join(f.content.split())[:settings.system2_catalog_content_chars]}"
                 for i, f in enumerate(bank, 1)]
        system = _CATALOG_SYSTEM.replace("{per_domain}", str(settings.system2_catalog_per_domain)).replace(
            "{total}", str(settings.system2_catalog_total))
        response = await self.factory.llm.chat_json(
            system=system, user="Recorded facts:\n" + "\n".join(lines), max_tokens=2048, temperature=0.0,
        )
        domains = response.get("domains") if isinstance(response, dict) else None
        if domains is None and isinstance(response, dict) and set(response) & set(CATALOG_DOMAINS):
            domains = response  # the domain map returned without its wrapper key
        if not isinstance(domains, dict):
            raise DecisionContextError(f"persona catalog must be a domains object, got {str(response)[:200]!r}")
        chosen: list[tuple[str, MemoryNode]] = []
        used: set[int] = set()
        for domain in CATALOG_DOMAINS:
            entries = domains.get(domain) or []
            if not isinstance(entries, list):
                continue
            taken = 0
            for entry in entries:
                if (not isinstance(entry, int) or isinstance(entry, bool) or not 1 <= entry <= len(bank)
                        or entry in used or taken >= settings.system2_catalog_per_domain
                        or len(chosen) >= settings.system2_catalog_total):
                    continue
                used.add(entry)
                taken += 1
                chosen.append((domain, bank[entry - 1]))
        if not chosen:
            return 0
        content = "\n".join(f"- [{domain}] {fact.content}" for domain, fact in chosen)
        [vector] = await self.factory.embed.embed_batch([content])
        evidence_ids = sorted(f.node_id for _, f in chosen)
        node = MemoryNode(
            node_id=_derived_id(scope, CATALOG_KIND, fingerprint, evidence_ids),
            content=content, layer=Layer.L3_DERIVED, kind=CATALOG_KIND,
            app_id=scope[0], user_id=scope[1], agent_id=scope[2],
            memory_at=max((f.memory_at for _, f in chosen if f.memory_at is not None), default=None),
            custom={"source_ids": evidence_ids, "fingerprint": fingerprint,
                    "source_node_id": min(evidence_ids)},
            supersedes=[c.node_id for c in current],
            embedding=vector,
        )
        await self._commit([node], [], scope, retire=current)
        return len(chosen)

    # ---- commit ---------------------------------------------------------------

    async def _commit(
        self, nodes: list[MemoryNode], batch: list[MemoryNode], scope: tuple[str, str, str],
        *, retire: list[MemoryNode] | None = None,
    ) -> None:
        async with self.factory.commit_locks.get("decision::" + "::".join(scope)):
            ok = await asyncio.to_thread(
                self.factory.vector.apply_version_plan,
                nodes=nodes,
                # Optimistic CAS on the memories this step reasoned over: if one was
                # superseded meanwhile, the plan is rejected rather than deriving from
                # stale evidence.
                expected={f.node_id: f.revision for f in [*batch, *(retire or [])]},
                archives=[],
                scope=scope,
                request_key="s2:" + ":".join(sorted([*(f.node_id for f in batch), *(n.node_id for n in nodes)])
                                             or ["empty"]),
                processed_evidence=sorted(f.node_id for f in batch),
            )
        if not ok:
            raise DecisionContextError("System2 commit raced with a concurrent write")
        await self._recover_projections(scope)

    async def _recover_projections(self, scope: tuple[str, str, str]) -> None:
        """Project every committed L3 plan of ``scope`` not yet in the graph."""
        if self.factory.graph is None:
            return
        for plan_id, node_ids in self.factory.vector.pending_projections(scope):
            nodes = self.factory.vector.get_by_ids(node_ids)
            if set(nodes) != set(node_ids):
                raise DecisionContextError("committed projection is missing a version")
            self._project(list(nodes.values()))
            self.factory.vector.acknowledge_projection(plan_id)

    def _project(self, nodes: list[MemoryNode]) -> None:
        """Graph is a rebuildable projection of committed schemas and their evidence."""
        graph = self.factory.graph
        for node in nodes:
            if node.layer != Layer.L3_DERIVED or node.kind not in SCHEMA_KINDS:
                continue
            graph.add_node(GraphNode(
                node_id=node.node_id, layer=node.layer.value, content=node.content,
                app_id=node.app_id, user_id=node.user_id, agent_id=node.agent_id,
                embedding=node.embedding or [], tags=node.tags,
                custom={**(node.custom or {}), "kind": node.kind},
                gmt_created=node.gmt_created,
            ))
            for fid in (node.custom or {}).get("source_ids", []):
                graph.add_evidence(schema_id=node.node_id, fact_id=fid)


__all__ = [
    "System2Agent", "SCHEMA_KIND", "CORE_SCHEMA_KIND", "CATALOG_KIND", "SCHEMA_KINDS",
    "CATALOG_DOMAINS", "cluster_facts", "collision_groups", "raw_source",
]
