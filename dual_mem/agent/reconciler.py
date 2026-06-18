# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Reconciler that recalls related existing memories, merges supersedes chains
and asks the LLM to emit ADD/DELETE ops integrating new memories losslessly.
"""
import json
from dataclasses import dataclass, field

from dual_mem.agent import prompts
from dual_mem.isolation import build_filter
from dual_mem.providers.embedding import EmbedService
from dual_mem.providers.llm import LLMClient
from dual_mem.storage.vector_store import VectorStore
from dual_mem.types import Layer, MemoryStatus


@dataclass
class ReconcileOp:
    """A single reconciliation operation (ADD or DELETE) parsed from the LLM output."""

    op: str = "ADD"
    content: str | None = None
    layer: str | None = None
    supersedes: list[str] = field(default_factory=list)
    supersede_reason: str = ""
    tags: list[str] = field(default_factory=list)
    memory_id: str | None = None
    reason: str = ""


class Reconciler:
    """Recalls candidates and drives the LLM to integrate new memories into the store."""

    SEARCH_THRESHOLD = 0.3
    SEARCH_TOPK = 20
    FINAL_TOPK = 10

    def __init__(
        self,
        *,
        llm: LLMClient,
        embed: EmbedService,
        vector: VectorStore,
        enable_search_query: bool = False,
    ):
        self.llm = llm
        self.embed = embed
        self.vector = vector
        self.enable_search_query = enable_search_query

    def reconcile(
        self,
        *,
        new_memories: list[str],
        new_memories_meta: list[dict],
        app_id: str,
        user_id: str,
        agent_id: str,
        current_time: str,
    ) -> list[ReconcileOp]:
        """Recall related memories and return LLM-proposed ADD/DELETE ops for the new batch."""
        if not new_memories:
            return []

        search_queries = (
            self._gen_search_queries(new_memories) if self.enable_search_query else []
        )

        where = build_filter(
            app_ids=[app_id],
            user_id=user_id,
            agent_ids=[agent_id],
            layers=[Layer.L2_FACT, Layer.L4_IDENTITY],
            statuses=[MemoryStatus.ACTIVE],
        )

        candidate_map: dict = {}
        candidate_scores: dict = {}
        for text in [*new_memories, *search_queries]:
            embedding = self.embed.embed(text)
            for node in self.vector.query(embedding=embedding, where=where, top_k=self.SEARCH_TOPK):
                if node.score < self.SEARCH_THRESHOLD:
                    continue
                nid = node.node_id
                if nid not in candidate_map:
                    candidate_map[nid] = node
                    candidate_scores[nid] = node.score
                else:
                    candidate_scores[nid] = max(candidate_scores[nid], node.score)

        if not candidate_map:
            return [
                ReconcileOp(
                    op="ADD",
                    content=meta.get("content", ""),
                    layer=meta.get("layer"),
                    supersedes=[],
                    tags=list(meta.get("tags") or []),
                )
                for meta in new_memories_meta
            ]

        chain_ancestors = self._merge_chains(candidate_map)

        head_ids = sorted(
            chain_ancestors.keys(),
            key=lambda hid: candidate_scores.get(hid, 0.0),
            reverse=True,
        )[: self.FINAL_TOPK * 2]

        existing_tags: set = set()
        for hid in head_ids:
            existing_tags.update(candidate_map[hid].tags)
            for anc in chain_ancestors[hid]:
                existing_tags.update(anc.tags)

        existing_mem_list = []
        for hid in sorted(head_ids, key=lambda h: candidate_map[h].memory_at or candidate_map[h].gmt_created):
            node = candidate_map[hid]
            item = {
                "memory_id": node.node_id,
                "content": node.content,
                "memory_at": node.memory_at,
                "layer": node.layer.value,
                "tags": list(node.tags),
            }
            ancestors = sorted(
                chain_ancestors[hid],
                key=lambda n: n.memory_at or n.gmt_created,
                reverse=True,
            )
            if ancestors:
                item["history_versions"] = [
                    {
                        "content": anc.content,
                        "memory_at": anc.memory_at,
                        "layer": anc.layer.value,
                        "tags": list(anc.tags),
                    }
                    for anc in ancestors
                ]
            existing_mem_list.append(item)

        new_mem_lines = "\n".join(
            self._format_new_memory(i, meta) for i, meta in enumerate(new_memories_meta)
        )
        existing_lines = json.dumps(existing_mem_list, ensure_ascii=False, indent=2)
        existing_tags_line = ", ".join(sorted(existing_tags)) if existing_tags else "(none yet)"

        joined = "\n".join(new_memories)
        system = prompts.pick(prompts.RECONCILE_ZH, prompts.RECONCILE_EN, joined).format(
            current_time=current_time,
            existing_memories=existing_lines,
            new_memories=new_mem_lines,
            existing_tags=existing_tags_line,
        )

        for _ in range(3):
            try:
                data = self.llm.chat_json(system=system, user=new_mem_lines)
            except json.JSONDecodeError:
                continue
            ops = self._parse_ops(data)
            if ops or data in ([], {}):
                return ops
        return []

    def _gen_search_queries(self, new_memories: list[str]) -> list[str]:
        """Ask the LLM for extra recall queries derived from the new memories."""
        mem_lines = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(new_memories))
        joined = "\n".join(new_memories)
        system = prompts.pick(prompts.SEARCH_QUERY_ZH, prompts.SEARCH_QUERY_EN, joined).format(
            new_memories=mem_lines
        )
        try:
            queries = self.llm.chat_json(system=system, user=mem_lines, json_object=False)
        except json.JSONDecodeError:
            return []
        if not isinstance(queries, list):
            return []
        return [q for q in queries if isinstance(q, str) and q.strip()]

    def _merge_chains(self, candidate_map: dict) -> dict:
        """Group candidate nodes under their chain head, collecting ancestors per head."""
        chain_ancestors: dict = {}
        for nid, node in candidate_map.items():
            superseded_by = node.superseded_by
            if superseded_by and superseded_by[-1] in candidate_map:
                head_id = superseded_by[-1]
                visited = set()
                while head_id in candidate_map:
                    nxt = candidate_map[head_id].superseded_by
                    if nxt and nxt[-1] in candidate_map and nxt[-1] not in visited:
                        visited.add(head_id)
                        head_id = nxt[-1]
                    else:
                        break
                chain_ancestors.setdefault(head_id, [])
                if nid != head_id:
                    chain_ancestors[head_id].append(node)
            else:
                chain_ancestors.setdefault(nid, [])
        return chain_ancestors

    @staticmethod
    def _format_new_memory(index: int, meta: dict) -> str:
        """Format one new memory (with optional tags) as a numbered prompt line."""
        line = f"{index + 1}. {meta.get('content', '')}"
        tags = meta.get("tags")
        if tags:
            line += f"\n   tags: {', '.join(str(t) for t in tags)}"
        return line

    @staticmethod
    def _parse_ops(data) -> list[ReconcileOp]:
        """Normalize the LLM's grouped/flat output into validated, conflict-free ReconcileOps."""
        if isinstance(data, dict):
            for key in ("updates", "groups", "results"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [data]
        if not isinstance(data, list):
            data = [data]

        flat_items: list[tuple] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            if "ops" in entry and isinstance(entry.get("ops"), list):
                group_reason = str(entry.get("reason") or "").strip()
                for op_dict in entry["ops"]:
                    if isinstance(op_dict, dict):
                        flat_items.append((op_dict, group_reason))
            elif "op" in entry:
                flat_items.append((entry, ""))

        parsed: list[ReconcileOp] = []
        for op_dict, group_reason in flat_items:
            op_type = str(op_dict.get("op", "")).upper()
            op_reason = str(op_dict.get("reason") or "").strip() or group_reason

            if op_type == "ADD":
                supersedes = op_dict.get("supersedes", [])
                if not isinstance(supersedes, list):
                    supersedes = [supersedes] if supersedes else []
                raw_tags = op_dict.get("tags", []) or []
                if not isinstance(raw_tags, list):
                    raw_tags = [raw_tags]
                tags = [str(t).strip().lower() for t in raw_tags if t and str(t).strip()][:3]
                parsed.append(
                    ReconcileOp(
                        op="ADD",
                        content=op_dict.get("content"),
                        layer=op_dict.get("layer"),
                        supersedes=[s for s in supersedes if s],
                        supersede_reason=str(op_dict.get("supersede_reason") or "").strip(),
                        tags=tags,
                        reason=op_reason,
                    )
                )
            elif op_type == "DELETE":
                mid = str(op_dict.get("memory_id") or "").strip()
                if not mid:
                    continue
                parsed.append(ReconcileOp(op="DELETE", memory_id=mid, reason=op_reason))

        touched: dict = {}
        validated: list[ReconcileOp] = []
        for op in parsed:
            conflict_ids: list[str] = []
            if op.op == "ADD" and op.supersedes:
                conflict_ids = list(op.supersedes)
            elif op.op == "DELETE" and op.memory_id:
                conflict_ids = [op.memory_id]

            if any(mid in touched for mid in conflict_ids):
                continue
            for mid in conflict_ids:
                touched[mid] = op.op
            validated.append(op)

        return validated
