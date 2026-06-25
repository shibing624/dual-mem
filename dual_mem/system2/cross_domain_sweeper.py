# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Cross-domain abstraction sweeper. Three-step design: (1) per-basic-schema
behavioral abstraction via LLM -> embed, (2) cosine-collision matrix + Union-Find clustering,
(3) higher-order LLM induction on each cluster to emit a core L6 schema linked via
CROSS_ABSTRACTS_TO.
"""
import asyncio
import json
import logging
import time
import uuid

import numpy as np

from dual_mem.agent import prompts
from dual_mem.providers.llm import is_chinese
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
from dual_mem.types import Layer

logger = logging.getLogger("dual_mem.system2.sweeper")


class _UnionFind:
    """Disjoint-set forest used to merge cosine-collision pairs into clusters."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        """Find the canonical root of x, applying path compression on the way."""
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        """Merge the sets containing x and y."""
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[ry] = rx

    def clusters(self) -> dict[str, list[str]]:
        """Group every known element under its root, producing the final clusters."""
        groups: dict[str, list[str]] = {}
        for x in self._parent:
            groups.setdefault(self.find(x), []).append(x)
        return groups


class CrossDomainSweeper:
    """Synthesizes higher-order core L6 schemas from cross-domain basic schemas."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    async def run(self, *, app_id: str, user_id: str, agent_id: str = "") -> dict:
        """Perform abstraction -> collision -> induction; return per-stage stats."""
        graph = self.factory.graph
        if graph is None:
            return {"triggered": False, "reason": "graph disabled"}

        all_schemas = graph.list_by_layer(
            layer=Layer.L6_SCHEMA.value, user_id=user_id, app_ids=[app_id]
        )
        basics = [n for n in all_schemas if (n.custom or {}).get("sub_type") != "core"]

        min_basics = self.factory.settings.cross_domain_min_basics
        if len(basics) < min_basics:
            return {"triggered": False, "basics_count": len(basics)}

        logger.info(
            "sweeper start user=%s basics=%d threshold=%d",
            user_id, len(basics), min_basics,
        )

        # Step 1: per-basic behavioral abstraction (cached via custom["behavior_abstraction"]).
        abstractions = await self._abstract_basics(basics)
        if not abstractions:
            return {"triggered": False, "reason": "abstraction failed"}

        # Step 2: cosine collision + Union-Find.
        clusters = self._collide(abstractions)
        if not clusters:
            logger.info("sweeper user=%s clusters=0 cores=0", user_id)
            return {"triggered": True, "clusters": 0, "cores": 0}

        # Step 3: higher-order induction per cluster (>=2 schemas).
        cores_created = 0
        for member_ids in clusters:
            if len(member_ids) < 2:
                continue
            cluster_schemas = [n for n in basics if n.node_id in set(member_ids)]
            core_id = await self._induce_core(cluster_schemas, app_id=app_id, user_id=user_id, agent_id=agent_id)
            if core_id is not None:
                cores_created += 1
        logger.info(
            "sweeper done user=%s clusters=%d cores=%d",
            user_id, len(clusters), cores_created,
        )
        return {"triggered": True, "clusters": len(clusters), "cores": cores_created}

    # ---- Step 1: per-basic behavioral abstraction --------------------------------------

    async def _abstract_basics(self, basics: list[GraphNode]) -> list[tuple[GraphNode, list[float]]]:
        """For each basic schema, produce a behavioral-abstraction embedding (cached).

        Cached nodes short-circuit immediately. Uncached nodes run their LLM abstraction
        step concurrently (``asyncio.gather``), then every freshly produced abstraction
        text is embedded in a single ``embed_batch`` round-trip instead of one serial
        ``embed`` call per node — collapsing N LLM RTTs + N embed RTTs into 1 concurrent
        LLM wave + 1 batched embed call.
        """
        llm = self.factory.llm
        embed = self.factory.embed
        graph = self.factory.graph
        if llm is None or graph is None:
            return []

        out: list[tuple[GraphNode, list[float]]] = []
        pending: list[GraphNode] = []
        for node in basics:
            cached = (node.custom or {}).get("behavior_embedding")
            if isinstance(cached, list) and cached:
                out.append((node, cached))
            else:
                pending.append(node)

        if not pending:
            return out

        async def _abstract_one(node: GraphNode) -> str | None:
            """Run the behavioral-abstraction LLM for one node; return the text or None."""
            system = (
                prompts.BEHAVIOR_ABSTRACTION_ZH
                if is_chinese(node.content)
                else prompts.BEHAVIOR_ABSTRACTION_EN
            ).format(content=node.content)
            try:
                parsed = await llm.chat_json(system=system, user=node.content)
            except json.JSONDecodeError:
                return None
            text = (parsed or {}).get("abstraction_for_embedding") if isinstance(parsed, dict) else None
            if not isinstance(text, str) or not text.strip():
                return None
            return text

        # Wave 1: concurrent LLM abstraction for all uncached nodes.
        texts = await asyncio.gather(*(_abstract_one(node) for node in pending))

        fresh_nodes: list[GraphNode] = []
        fresh_texts: list[str] = []
        for node, text in zip(pending, texts):
            if text is None:
                continue
            fresh_nodes.append(node)
            fresh_texts.append(text)

        if not fresh_texts:
            return out

        # Wave 2: one batched embedding call for every fresh abstraction text.
        embeddings = await embed.embed_batch(fresh_texts)

        # Cache each behavior embedding back onto its graph node so subsequent runs skip
        # the LLM call. Re-add the node with merged custom payload.
        for node, text, embedding in zip(fresh_nodes, fresh_texts, embeddings):
            new_custom = dict(node.custom or {})
            new_custom["behavior_abstraction"] = text
            new_custom["behavior_embedding"] = embedding
            node.custom = new_custom
            graph.add_node(node)
            out.append((node, embedding))
        return out

    # ---- Step 2: cosine collision + Union-Find -----------------------------------------

    def _collide(
        self, abstractions: list[tuple[GraphNode, list[float]]]
    ) -> list[list[str]]:
        """Pairwise cosine collisions above threshold are merged via Union-Find clustering."""
        if len(abstractions) < 2:
            return []
        threshold = self.factory.settings.cross_domain_threshold
        ids = [item[0].node_id for item in abstractions]
        matrix = np.array([item[1] for item in abstractions], dtype=float)
        # Cosine similarity = normalized dot product.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = matrix / norms
        sim = normed @ normed.T

        uf = _UnionFind()
        for i in range(len(ids)):
            uf.find(ids[i])  # ensure singletons appear in clusters too
            for j in range(i + 1, len(ids)):
                if float(sim[i, j]) >= threshold:
                    uf.union(ids[i], ids[j])
        groups = uf.clusters()
        return [members for members in groups.values() if len(members) >= 2]

    # ---- Step 3: higher-order induction ------------------------------------------------

    async def _induce_core(
        self,
        cluster: list[GraphNode],
        *,
        app_id: str,
        user_id: str,
        agent_id: str,
    ) -> str | None:
        """Ask the LLM for a higher-order pattern; create the core node + CROSS_ABSTRACTS_TO edges."""
        llm = self.factory.llm
        graph = self.factory.graph
        if llm is None or graph is None or not cluster:
            return None
        patterns_text = "\n".join(f"- {n.content}  (id={n.node_id})" for n in cluster)
        joined = " ".join(n.content for n in cluster)
        system = (
            prompts.CROSS_DOMAIN_INDUCTION_ZH
            if is_chinese(joined)
            else prompts.CROSS_DOMAIN_INDUCTION_EN
        ).format(patterns=patterns_text)
        try:
            parsed = await llm.chat_json(system=system, user=patterns_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        core_pattern = parsed.get("core_pattern")
        if not isinstance(core_pattern, str) or not core_pattern.strip():
            return None

        member_ids = {n.node_id for n in cluster}
        targets = [
            sid for sid in (parsed.get("schema_ids") or []) if isinstance(sid, str) and sid in member_ids
        ]
        if not targets:
            targets = sorted(member_ids)

        core_id = str(uuid.uuid4())
        core_embedding = await self.factory.embed.embed(core_pattern)
        graph.add_node(
            GraphNode(
                node_id=core_id,
                layer=Layer.L6_SCHEMA.value,
                content=core_pattern,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                embedding=core_embedding,
                custom={
                    "sub_type": "core",
                    "reasoning": parsed.get("reasoning", ""),
                    "confidence": parsed.get("confidence", 0.0),
                },
                gmt_created=int(time.time()),
            )
        )
        for basic_id in targets:
            graph.add_edge(from_id=basic_id, to_id=core_id, rel="CROSS_ABSTRACTS_TO")
        return core_id
