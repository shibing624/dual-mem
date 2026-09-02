# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)

记忆调和器（Reconciler）—— System1 写入链路的「进化链」决策模块。

## 在整体流水线中的位置

Extractor 快写 L2/L4 之后，新记忆可能与库里已有条目重复、冲突或构成时序更新。
Reconciler 负责**召回相关旧记忆 → 让 LLM 判断关系 → 输出 ADD/DELETE 操作列表**，
由下游（MemAgent 同步路径或显式 digest 调用的 ReconcilerWorker）落盘并维护
``supersedes`` / ``superseded_by`` 进化链指针。

两条调用路径：
  - ``reconcile_sync=True``：``MemAgent`` 在写入路径内同步调用（强一致，多 ~1 次 LLM）
  - 默认延迟处理：``ReconcilerWorker`` 在显式 ``digest()`` 时处理 reconcile 队列

## 核心流程（``Reconciler.reconcile``）

1. **向量召回**：对新记忆（+ 可选 LLM 改写查询）做 embed，在 L2/L4 ACTIVE 层检索候选
2. **进化链合并**：``_merge_chains`` 把候选按链头分组，附带 ``history_versions``
3. **LLM 决策**：中英 prompt（``RECONCILE_ZH/EN``）输入新旧记忆，输出 JSON ops
4. **解析校验**：``_parse_ops`` 归一化 LLM 输出，去重冲突的 supersedes/DELETE

## 输出语义（``ReconcileOp``）

- ``ADD``：写入整合后的内容；``supersedes`` 指向被取代的旧节点 ID
- ``DELETE``：软删（下游标 SHADOW / is_latest=False）
- ``update_type``：OVERRIDE / MERGE / SUPPLEMENT / TEMPORAL / NEGATE / CONFLICT，
  供下游写入 ``node.custom``（如 TEMPORAL 的 ``temporal_scope``、NEGATE 的 ``negation``、
  MERGE 的 ``merged_timestamps``）

本模块只做**决策**，不直接写库；向量检索阈值见 ``SEARCH_THRESHOLD`` / ``FINAL_TOPK``。
"""
import json
import logging
import asyncio
from dataclasses import dataclass, field

from dual_mem.agent import prompts
from dual_mem.isolation import build_filter
from dual_mem.providers.embedding import EmbedService
from dual_mem.providers.llm import LLMClient
from dual_mem.storage.vector_store import VectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus

_VALID_UPDATE_TYPES = frozenset({
    "OVERRIDE", "SUPPLEMENT", "TEMPORAL", "NEGATE", "CONFLICT", "MERGE", "",
})

logger = logging.getLogger("dual_mem.agent.reconcile")


@dataclass
class ReconcileOp:
    """A single reconciliation operation parsed from the LLM output.

    update_type classifies the relationship between the new memory and existing memories
    (OVERRIDE / MERGE / SUPPLEMENT / TEMPORAL / NEGATE / CONFLICT), so downstream apply logic can
    persist temporal_scope on TEMPORAL ops and tag NEGATE ops in node.custom for the reader.
    """

    op: str = "ADD"
    content: str | None = None
    layer: str | None = None
    supersedes: list[str] = field(default_factory=list)
    supersede_reason: str = ""
    tags: list[str] = field(default_factory=list)
    memory_id: str | None = None
    reason: str = ""
    update_type: str = ""           # OVERRIDE | MERGE | SUPPLEMENT | TEMPORAL | NEGATE | CONFLICT | ""
    temporal_scope: str | None = None
    negation: bool = False
    new_index: int | None = None    # 1-based index into the new-memory batch (SKIP)


def fold_absorb_deletes(ops: list[ReconcileOp]) -> list[ReconcileOp]:
    """Rewrite Objective-2 ADD+DELETE absorb into a MERGE supersedes chain.

    Complementary merge used to SHADOW the old node via DELETE. That is the
    destructive absorb this function removes: the DELETE ids become ``supersedes``
    on an ADD, ``update_type`` becomes MERGE when it was empty/SUPPLEMENT, and
    those DELETE ops are dropped. Orphan DELETE with no ADD is kept (exact dup).
    """
    adds = [op for op in ops if op.op == "ADD"]
    deletes = [op for op in ops if op.op == "DELETE" and op.memory_id]
    others = [op for op in ops if op.op not in ("ADD", "DELETE")]
    if not deletes:
        return list(ops)
    covered = {sid for add in adds for sid in add.supersedes}
    orphans = [d for d in deletes if d.memory_id not in covered]
    kept_deletes: list[ReconcileOp] = []
    if orphans and adds:
        target = next((add for add in adds if not add.supersedes), adds[0])
        for doomed in orphans:
            if doomed.memory_id not in target.supersedes:
                target.supersedes.append(doomed.memory_id)
        if target.update_type in ("", "SUPPLEMENT"):
            target.update_type = "MERGE"
    elif orphans:
        kept_deletes = orphans
    return adds + others + kept_deletes


def _norm_skip_content(text: str | None) -> str:
    return " ".join((text or "").split()).lower()


def _is_empty_verdict(data) -> bool:
    """True when the LLM parsed cleanly and said there is no increment."""
    if data in ([], {}):
        return True
    if isinstance(data, dict):
        for key in ("updates", "groups", "results"):
            if key in data and data[key] == []:
                return True
    return False


def resolve_skip_targets(
    ops: list[ReconcileOp],
    *,
    new_node_ids: list[str],
    new_contents: list[str],
) -> list[str]:
    """Map SKIP ops onto fast-write node ids. Untargeted SKIP retracts the whole batch."""
    skips = [op for op in ops if op.op == "SKIP"]
    if not skips:
        return []
    id_by_content = {
        _norm_skip_content(content): nid
        for nid, content in zip(new_node_ids, new_contents)
    }
    allowed = set(new_node_ids)
    targets: set[str] = set()
    for op in skips:
        if not op.memory_id and op.new_index is None and not op.content:
            return list(new_node_ids)
        if op.memory_id and op.memory_id in allowed:
            targets.add(op.memory_id)
        if op.new_index is not None:
            index = op.new_index - 1 if op.new_index >= 1 else op.new_index
            if 0 <= index < len(new_node_ids):
                targets.add(new_node_ids[index])
        if op.content:
            matched = id_by_content.get(_norm_skip_content(op.content))
            if matched:
                targets.add(matched)
    return list(targets)


def collect_merged_timestamps(
    *,
    update_type: str,
    new_memory_at: int | None,
    old_nodes: list[MemoryNode],
    extra: list[int] | None = None,
) -> list[int]:
    """Union of session/node timestamps for complementary MERGE only."""
    if update_type != "MERGE":
        return []
    stamps: list[int] = []
    for ts in (new_memory_at, *(extra or [])):
        if ts is not None:
            stamps.append(int(ts))
    for old in old_nodes:
        if old.memory_at is not None:
            stamps.append(int(old.memory_at))
        prev = (old.custom or {}).get("merged_timestamps")
        if isinstance(prev, list):
            stamps.extend(int(x) for x in prev if isinstance(x, (int, float)))
    return sorted(set(stamps))


def reflect_custom(
    op: ReconcileOp,
    *,
    merged_timestamps: list[int] | None = None,
) -> dict:
    """Pack reflector metadata into node.custom."""
    out: dict = {}
    if op.update_type:
        out["update_type"] = op.update_type
    if op.temporal_scope:
        out["temporal_scope"] = op.temporal_scope
    if op.negation:
        out["negation"] = True
    if merged_timestamps:
        out["merged_timestamps"] = list(merged_timestamps)
    return out


def build_apply_custom(
    op: ReconcileOp,
    *,
    vector,
    new_memory_at: int | None,
    extra: list[int] | None = None,
    source_node_id: str = "",
) -> dict:
    """Build node.custom for an applied ADD, including MERGE timestamp union."""
    old_nodes: list[MemoryNode] = []
    if op.update_type == "MERGE":
        for old_id in op.supersedes:
            old = vector.get(old_id)
            if old is not None:
                old_nodes.append(old)
    custom = reflect_custom(
        op,
        merged_timestamps=collect_merged_timestamps(
            update_type=op.update_type,
            new_memory_at=new_memory_at,
            old_nodes=old_nodes,
            extra=extra,
        ),
    )
    if source_node_id:
        custom["source_node_id"] = source_node_id
    return custom


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
        policy: str = "balanced",
        weak_candidate_score: float = 0.5,
        non_destructive: bool = False,
        prompt_pair: tuple[str, str] | None = None,
        use_hy_contract: bool = False,
    ):
        self.llm = llm
        self.embed = embed
        self.vector = vector
        self.enable_search_query = enable_search_query
        self.policy = policy
        self.weak_candidate_score = weak_candidate_score
        # Optional (zh, en) prompt override, e.g. the System1 hy-style contract.
        self.prompt_pair = prompt_pair
        # non_destructive: reconcile 只做增量 ADD，永不 supersede/DELETE
        # 原始 fact。开启后 _apply_non_destructive 在 _parse_ops 输出上硬剥离 supersedes
        # 和 DELETE，保证记忆库只增不减，计数/聚合类问题零丢事实风险。
        self.non_destructive = non_destructive
        # use_hy_contract: 使用 hy_memory 的 reconcile 契约 — bare JSON 数组，
        # ADD/DELETE op，无 reason 分组。当 True 时会覆盖 prompt_pair。
        self.use_hy_contract = use_hy_contract

    async def reconcile(
        self,
        *,
        new_memories: list[str],
        new_memories_meta: list[dict],
        app_id: str,
        user_id: str,
        agent_id: str,
        current_time: str,
        exclude_ids: list[str] | None = None,
    ) -> list[ReconcileOp]:
        """Recall related memories and return LLM-proposed ADD/DELETE ops for the new batch.

        ``exclude_ids`` are dropped from the candidate set — used by the async reconcile
        worker to keep the just-written fast-write originals (which are still ACTIVE) from
        coming back as near-1.0 self-matches and polluting the LLM's "existing" list.
        """
        if not new_memories:
            return []

        excluded = set(exclude_ids or ())

        search_queries = (
            await self._gen_search_queries(new_memories) if self.enable_search_query else []
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
        recall_texts = [*new_memories, *search_queries]
        if recall_texts:
            embeddings = await self.embed.embed_batch(recall_texts)

            async def _query_hits(embedding: list[float]) -> list:
                return await asyncio.to_thread(
                    self.vector.query,
                    embedding=embedding,
                    where=where,
                    top_k=self.SEARCH_TOPK,
                )

            hit_lists = await asyncio.gather(
                *[_query_hits(emb) for emb in embeddings],
            )
            for nodes in hit_lists:
                for node in nodes:
                    if node.score < self.SEARCH_THRESHOLD:
                        continue
                    nid = node.node_id
                    if nid in excluded:
                        continue
                    if nid not in candidate_map:
                        candidate_map[nid] = node
                        candidate_scores[nid] = node.score
                    else:
                        candidate_scores[nid] = max(candidate_scores[nid], node.score)

        # Fast path: no candidate, or the strongest candidate is too weak to be a real
        # conflict. Skip the LLM entirely and keep every new memory as an independent
        # SUPPLEMENT (zero merge risk, saves one reconcile LLM call on the common no-conflict
        # case where the batch is genuinely new facts).
        best_score = max(candidate_scores.values(), default=0.0)
        if not candidate_map or best_score < self.weak_candidate_score:
            return [
                ReconcileOp(
                    op="ADD",
                    content=meta.get("content", ""),
                    layer=meta.get("layer"),
                    supersedes=[],
                    tags=list(meta.get("tags") or []),
                    update_type="SUPPLEMENT",
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
        if self.use_hy_contract:
            # hy contract: EN-only, bare JSON array, ADD/DELETE ops
            system = prompts.RECONCILE_HY_EN.format(
                current_time=current_time,
                existing_memories=existing_lines,
                new_memories=new_mem_lines,
            )
            logger.info("[reconciler] hy_contract prompt (EN only, %d chars)", len(system))
        else:
            prompt_zh, prompt_en = self.prompt_pair or (prompts.RECONCILE_ZH, prompts.RECONCILE_EN)
            system = prompts.pick(prompt_zh, prompt_en, joined).format(
                current_time=current_time,
                existing_memories=existing_lines,
                new_memories=new_mem_lines,
                existing_tags=existing_tags_line,
            )
        if self.policy == "conservative" and not self.use_hy_contract:
            system += prompts.pick(
                prompts.RECONCILE_POLICY_CONSERVATIVE_ZH,
                prompts.RECONCILE_POLICY_CONSERVATIVE_EN,
                joined,
            )

        for _ in range(3):
            try:
                data = await self.llm.chat_json(
                    system=system,
                    user=new_mem_lines,
                    max_tokens=8192,
                    json_array=self.use_hy_contract,
                )
            except json.JSONDecodeError as exc:
                logger.warning("reconcile JSON parse failed, retrying: %s", exc)
                continue
            ops = self._parse_ops(data)
            if not self.non_destructive and self.policy != "conservative":
                ops = fold_absorb_deletes(ops)
            if self.non_destructive:
                ops = self._apply_non_destructive(ops)
            if ops:
                logger.debug(
                    "reconcile candidates=%d ops=%d supersedes=%d skip=%d",
                    len(candidate_map), len(ops),
                    sum(1 for op in ops if op.op == "ADD" and op.supersedes),
                    sum(1 for op in ops if op.op == "SKIP"),
                )
                return ops
            if _is_empty_verdict(data):
                logger.info("reconcile empty verdict → SKIP all new memories")
                return [ReconcileOp(op="SKIP")]
        logger.warning("reconcile produced no parseable ops after 3 retries")
        return []

    def _apply_non_destructive(self, ops: list[ReconcileOp]) -> list[ReconcileOp]:
        """Strip all supersedes/DELETE — make reconcile purely additive.

        Original facts stay ACTIVE forever; the reconciler may still ADD enriched/summary
        memories alongside originals, but never modifies or removes existing ones. This
        guarantees zero fact-loss on counting/aggregation questions where balanced mode
        would merge distinct events into one summary and drop amounts.
        """
        result: list[ReconcileOp] = []
        stripped = 0
        dropped = 0
        for op in ops:
            if op.op == "DELETE":
                dropped += 1
                continue
            if op.op == "SKIP":
                result.append(op)
                continue
            if op.supersedes:
                stripped += 1
                op.supersedes = []
                op.supersede_reason = ""
                op.update_type = "SUPPLEMENT"
            result.append(op)
        if stripped or dropped:
            logger.info(
                "non_destructive reconcile: stripped %d supersedes, dropped %d DELETEs",
                stripped,
                dropped,
            )
        return result

    async def _gen_search_queries(self, new_memories: list[str]) -> list[str]:
        """Ask the LLM for extra recall queries derived from the new memories."""
        mem_lines = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(new_memories))
        joined = "\n".join(new_memories)
        system = prompts.pick(prompts.SEARCH_QUERY_ZH, prompts.SEARCH_QUERY_EN, joined).format(
            new_memories=mem_lines
        )
        try:
            queries = await self.llm.chat_json(system=system, user=mem_lines, json_object=False)
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
        node_id = meta.get("node_id")
        if node_id:
            line += f"\n   node_id: {node_id}"
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
                update_type = str(op_dict.get("update_type") or "").strip().upper()
                if update_type not in _VALID_UPDATE_TYPES:
                    update_type = ""
                supersedes_ids = [s for s in supersedes if s]
                if update_type == "MERGE" and not supersedes_ids:
                    update_type = "SUPPLEMENT"
                temporal_scope = op_dict.get("temporal_scope")
                if temporal_scope is not None and not isinstance(temporal_scope, str):
                    temporal_scope = str(temporal_scope)
                parsed.append(
                    ReconcileOp(
                        op="ADD",
                        content=op_dict.get("content"),
                        layer=op_dict.get("layer"),
                        supersedes=supersedes_ids,
                        supersede_reason=str(op_dict.get("supersede_reason") or "").strip(),
                        tags=tags,
                        reason=op_reason,
                        update_type=update_type,
                        temporal_scope=temporal_scope or None,
                        negation=update_type == "NEGATE",
                    )
                )
            elif op_type == "DELETE":
                mid = str(op_dict.get("memory_id") or "").strip()
                if not mid:
                    continue
                parsed.append(ReconcileOp(op="DELETE", memory_id=mid, reason=op_reason))
            elif op_type == "SKIP":
                mid = str(op_dict.get("memory_id") or op_dict.get("node_id") or "").strip()
                raw_index = op_dict.get("new_index")
                new_index = None
                if raw_index is not None:
                    try:
                        new_index = int(raw_index)
                    except (TypeError, ValueError):
                        new_index = None
                content = str(op_dict.get("content") or "").strip()
                parsed.append(
                    ReconcileOp(
                        op="SKIP",
                        memory_id=mid or None,
                        content=content or None,
                        new_index=new_index,
                        reason=op_reason,
                    )
                )

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
