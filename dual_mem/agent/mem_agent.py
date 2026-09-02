# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MemAgent orchestrating the System1 cognitive pipeline. It fast-writes extracted
L2/L4 memories and, in dual mode, queues reconcile work for explicit digest(). When
reconcile_sync=True it reconciles inline for strong consistency. It also handles L7 intentions,
L0 basic profiles, and optional L3 summaries.
"""
import asyncio
import logging
import math
import re
import time
import unicodedata
import uuid
from datetime import datetime
from typing import Any

from dual_mem.agent import prompts
from dual_mem.agent.basic_profile import BasicProfileTool, normalize_basic_info
from dual_mem.agent.extractor import Extractor
from dual_mem.agent.reconciler import ReconcileOp, Reconciler, build_apply_custom
from dual_mem.agent.summarizer import Summarizer
from dual_mem.registry import ComponentFactory
from dual_mem.sdk_models import ChatMessage, CommitResult
from dual_mem.storage.graph_store import GraphNode
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.agent.mem")


class MemAgent:
    """Coordinates extractor, reconciler and summarizer for one input memory."""

    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory
        self.settings = factory.settings
        self.vector = factory.vector
        self.embed = factory.embed
        self.history = factory.history
        self.cache = factory.cache
        llm = factory.llm
        assert llm is not None, "MemAgent requires factory.llm (system1/dual mode)"
        self.basic_profile_tool = BasicProfileTool(vector=self.vector, embed=self.embed)
        cpt = self.settings.chars_per_token
        self.extractor = Extractor(
            llm=llm,
            max_content_chars=int(self.settings.extract_max_content_tokens * cpt),
            retry_on_failure=self.settings.extract_retry_on_failure,
            few_shot_enabled=self.settings.extract_few_shot_enabled,
        )
        self.summarizer = Summarizer(
            llm=llm,
            min_content_length=int(self.settings.summarizer_min_content_tokens * cpt),
        )
        self.reconciler = Reconciler(
            llm=llm,
            embed=self.embed,
            vector=self.vector,
            enable_search_query=self.settings.reconcile_search_query,
            policy=self.settings.reconcile_policy,
            weak_candidate_score=self.settings.reconcile_weak_candidate_score,
            non_destructive=self.settings.reconcile_non_destructive,
            use_hy_contract=self.settings.reconcile_use_hy_contract,
        )

    async def run_system1(
        self,
        *,
        content: str,
        raw_node: MemoryNode,
        messages: list[ChatMessage] | None = None,
        request_id: str = "",
    ) -> tuple[list[str], CommitResult, bool]:
        """Extract searchable memories, each embedded on its own content."""
        current_time = (
            datetime.fromtimestamp(raw_node.memory_at).isoformat(timespec="seconds")
            if raw_node.memory_at
            else ""
        )
        extracted = await self.extractor.extract_system1(
            content=content,
            current_time=current_time,
            current_messages=messages,
        )
        for key in ("identity", "facts", "intentions"):
            extracted[key] = self._normalize_memory_items(extracted.get(key))

        new_memories, new_meta = self._collect_new_memories(extracted)
        # System1 keeps intentions as first-class retrievable memories (same
        # pool as facts, tagged memory_type="intention"), mirroring the
        # reference reader where intention nodes rank alongside facts.
        seen_contents = {text.strip() for text in new_memories}
        for item in extracted.get("intentions") or []:
            if not isinstance(item, dict):
                continue
            intention_content = item.get("content")
            if not isinstance(intention_content, str) or not intention_content.strip():
                continue
            normalized = intention_content.strip()
            if normalized in seen_contents:
                continue
            seen_contents.add(normalized)
            new_memories.append(normalized)
            new_meta.append(
                {
                    "content": normalized,
                    "layer": "L2_FACT",
                    "memory_type": "intention",
                    "tags": item.get("tags") or [],
                    "speculate": item.get("speculate"),
                    "owner": item.get("owner", ""),
                }
            )
        is_ephemeral = bool(extracted.get("is_ephemeral", False))
        if is_ephemeral:
            commit_result = CommitResult(False, "extractor marked content ephemeral")
        elif not new_memories and not normalize_basic_info(extracted.get("basic_info")):
            commit_result = CommitResult(False, "extractor produced no persistable memory")
        else:
            commit_result = CommitResult(True, "extractor produced persistable memory")
        try:
            self.cache.log_pipeline(
                request_id=request_id,
                stage="EXTRACT",
                payload={
                    "n_identity": len(extracted.get("identity", [])),
                    "n_facts": len(extracted.get("facts", [])),
                    "n_intentions": len(extracted.get("intentions", [])),
                    "has_basic_info": bool(normalize_basic_info(extracted.get("basic_info"))),
                    "is_ephemeral": is_ephemeral,
                    "commit_passed": commit_result.passed,
                    "commit_reason": commit_result.reason,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("log_pipeline(EXTRACT) failed (system1): %s", exc)
        if not commit_result.passed:
            return [], commit_result, is_ephemeral

        settings = self.factory.settings
        if settings.system1_reconcile_enabled and not settings.reconcile_skip_llm:
            # Merge the chunk's new memories against the user's existing ACTIVE
            # memories (hy-style): candidate recall reuses freshly embedded new
            # contents; the reconcile LLM is skipped when no real conflict
            # candidate exists (fast path = plain ADD of every new memory).
            reconciler = Reconciler(
                llm=self.factory.llm,
                embed=self.embed,
                vector=self.vector,
                enable_search_query=False,
                policy=settings.reconcile_policy,
                weak_candidate_score=settings.reconcile_weak_candidate_score,
                non_destructive=False,
                prompt_pair=(prompts.RECONCILE_SYSTEM1_ZH, prompts.RECONCILE_SYSTEM1_EN),
            )
            ops = await reconciler.reconcile(
                new_memories=new_memories,
                new_memories_meta=new_meta,
                app_id=raw_node.app_id,
                user_id=raw_node.user_id,
                agent_id=raw_node.agent_id,
                current_time=current_time,
                exclude_ids=[raw_node.node_id],
            )
            if not ops:
                # "No evolution" verdict: the new memories still carry new
                # information, so they must be written as plain ADDs.
                ops = [
                    ReconcileOp(
                        op="ADD",
                        content=memory,
                        layer=meta.get("layer") or "L2_FACT",
                        tags=meta.get("tags") or [],
                        update_type="SUPPLEMENT",
                    )
                    for memory, meta in zip(new_memories, new_meta, strict=True)
                ]
            try:
                self.cache.log_pipeline(
                    request_id=request_id,
                    stage="RECONCILE",
                    payload={
                        "n_ops": len(ops),
                        "n_add": sum(1 for op in ops if op.op == "ADD"),
                        "n_delete": sum(1 for op in ops if op.op == "DELETE"),
                        "n_supersede_links": sum(len(op.supersedes) for op in ops),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("log_pipeline(RECONCILE) failed (system1): %s", exc)
        else:
            ops = [
                ReconcileOp(
                    op="ADD",
                    content=memory,
                    layer=meta.get("layer") or "L2_FACT",
                    tags=meta.get("tags") or [],
                    update_type="SUPPLEMENT",
                )
                for memory, meta in zip(new_memories, new_meta, strict=True)
            ]
        written_ids = await self._apply_system1_ops(ops=ops, raw_node=raw_node, new_meta=new_meta)
        return written_ids, commit_result, is_ephemeral

    async def _apply_system1_ops(
        self,
        *,
        ops: list[ReconcileOp],
        raw_node: MemoryNode,
        new_meta: list[dict[str, Any]],
    ) -> list[str]:
        """Apply reconcile ops for one System1 chunk with batched embedding/upsert."""
        meta_by_content = {str(meta.get("content", "")).strip(): meta for meta in new_meta}
        add_ops = [op for op in ops if op.op == "ADD" and op.content.strip()]
        nodes: list[MemoryNode] = []
        if add_ops:
            embeddings = await self.embed.embed_batch([op.content for op in add_ops])
            for op, embedding in zip(add_ops, embeddings, strict=True):
                layer_name = (op.layer or "L2_FACT").upper()
                try:
                    layer = Layer(layer_name)
                except ValueError:
                    layer = Layer.L2_FACT
                src_meta = meta_by_content.get(op.content.strip(), {})
                nodes.append(
                    MemoryNode(
                        content=op.content,
                        layer=layer,
                        app_id=raw_node.app_id,
                        user_id=raw_node.user_id,
                        agent_id=raw_node.agent_id,
                        session_id=raw_node.session_id,
                        status=MemoryStatus.ACTIVE,
                        is_latest=True,
                        supersedes=list(op.supersedes),
                        memory_at=raw_node.memory_at,
                        tags=op.tags or [],
                        custom={
                            **build_apply_custom(
                                op,
                                vector=self.vector,
                                new_memory_at=raw_node.memory_at,
                                extra=[raw_node.memory_at] if raw_node.memory_at is not None else None,
                                source_node_id=raw_node.node_id,
                            ),
                            "memory_type": src_meta.get("memory_type", "fact"),
                        },
                        embedding=embedding,
                    )
                )
        if nodes:
            await asyncio.to_thread(self.vector.upsert, nodes)
        for op, new_node in zip(add_ops, nodes, strict=True):
            for old_id in op.supersedes:
                if old_id != new_node.node_id:
                    await asyncio.to_thread(
                        self.vector.mark_superseded, old_id, superseded_by_id=new_node.node_id
                    )
        # DELETE ops are intentionally not applied in System1: supersede keeps
        # history reachable through evolution chains, but DELETE would drop
        # information irrecoverably — the reference system never deletes either.
        n_delete_skipped = sum(1 for op in ops if op.op == "DELETE")
        if n_delete_skipped:
            logger.info("system1 reconcile: skipped %d DELETE op(s)", n_delete_skipped)
        return [node.node_id for node in nodes]

    async def run(
        self,
        *,
        content: str,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        request_id: str,
        memory_at: int | None,
        messages: list[ChatMessage] | None = None,
        source_node_id: str = "",
    ) -> tuple[list[str], CommitResult, bool]:
        """Run the System1 pipeline for one raw memory.

        Returns ``(extra_node_ids, commit_result, is_ephemeral)``. The Extractor output is
        the only commit decision: ephemeral or empty structured output stays L1-only.
        """
        current_time = (
            datetime.fromtimestamp(memory_at).isoformat(timespec="seconds") if memory_at else ""
        )

        summary_task = self._begin_summarize_task(
            content=content,
            current_time=current_time,
        )
        extracted = await self.extractor.extract(
            content=content,
            current_time=current_time,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            current_messages=messages,
            history_messages=None,
        )

        for key in ("identity", "facts", "intentions"):
            extracted[key] = self._normalize_memory_items(extracted.get(key))
            if self.settings.canonical_dedup_enabled:
                extracted[key] = self._dedup_memory_items(extracted[key])
        is_ephemeral = bool(extracted.get("is_ephemeral"))
        basic_info = normalize_basic_info(extracted.get("basic_info"))
        extracted["basic_info"] = basic_info
        has_basic_info = bool(basic_info)
        has_persistable_output = bool(
            extracted["identity"]
            or extracted["facts"]
            or extracted["intentions"]
            or has_basic_info
        )
        if is_ephemeral:
            commit_result = CommitResult(
                passed=False,
                reason="extractor marked content ephemeral",
            )
        elif not has_persistable_output:
            commit_result = CommitResult(
                passed=False,
                reason="extractor produced no persistable memory",
            )
        else:
            commit_result = CommitResult(
                passed=True,
                reason="extractor produced persistable memory",
            )

        try:
            self.cache.log_pipeline(
                request_id=request_id,
                stage="EXTRACT",
                payload={
                    "n_identity": len(extracted.get("identity") or []),
                    "n_facts": len(extracted.get("facts") or []),
                    "n_intentions": len(extracted.get("intentions") or []),
                    "has_basic_info": has_basic_info,
                    "is_ephemeral": is_ephemeral,
                    "commit_passed": commit_result.passed,
                    "commit_reason": commit_result.reason,
                },
            )
        except Exception:
            logger.warning("Failed to log extract pipeline result", exc_info=True)
        logger.info(
            "extract identity=%d facts=%d intentions=%d ephemeral=%s",
            len(extracted.get("identity") or []),
            len(extracted.get("facts") or []),
            len(extracted.get("intentions") or []),
            is_ephemeral,
        )
        if not commit_result.passed:
            await self._cancel_summarize_task(
                summary_task,
                reason=commit_result.reason,
            )
            return [], commit_result, is_ephemeral

        new_memories, new_meta = self._collect_new_memories(extracted)
        basic_info_present = has_basic_info

        stored_ids: list[str] = []

        # ---- Step 3: Persist L0/L2/L4 nodes ---------------------------------------------
        # reconcile_sync writes L0/L2/L4 first then reconciles inline (strong consistency);
        # the default dual path fast-writes and leaves L2/L4 work for explicit digest().
        # L0 evolves via its own supersede chain and never enters that queue.
        if new_memories or basic_info_present:
            if self.settings.reconcile_sync:
                l0_ids, _ = await self._fast_write(
                    extracted,
                    [],
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    memory_at=memory_at,
                    source_node_id=source_node_id,
                )
                if new_memories:
                    ops = await self.reconciler.reconcile(
                        new_memories=new_memories,
                        new_memories_meta=new_meta,
                        app_id=app_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        current_time=current_time,
                    )
                    l2l4_ids = await self._apply_ops(
                        ops,
                        app_id=app_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        memory_at=memory_at,
                        source_node_id=source_node_id,
                    )
                else:
                    l2l4_ids = []
                stored_ids = l0_ids + l2l4_ids
            else:
                l0_ids, l2l4_ids = await self._fast_write(
                    extracted,
                    new_meta,
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    memory_at=memory_at,
                    source_node_id=source_node_id,
                )
                stored_ids = l0_ids + l2l4_ids
                if l2l4_ids and self.settings.mode == "dual":
                    self.cache.enqueue_reconcile_task(
                        app_id=app_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        node_ids=l2l4_ids,
                    )
        if stored_ids:
            try:
                self.cache.log_pipeline(
                    request_id=request_id,
                    stage="FAST_WRITE",
                    payload={
                        "node_ids": list(stored_ids),
                        "n_nodes": len(stored_ids),
                        "reconcile_sync": self.settings.reconcile_sync,
                    },
                )
            except Exception:
                logger.warning("Failed to log fast-write pipeline result", exc_info=True)

        try:
            summary = await summary_task if summary_task is not None else None
        except Exception as exc:
            logger.warning("summary task failed, continuing without L3: %s", exc)
            summary = None

        # ---- Step 4: L7 intentions + L3 summary (one embed batch when possible) ------
        intention_items = [
            intent
            for intent in (extracted.get("intentions") or [])
            if isinstance(intent, dict)
            and isinstance(intent.get("content"), str)
            and intent.get("content", "").strip()
        ]
        tail_texts: list[str] = []
        if summary:
            tail_texts.append(summary)
        tail_texts.extend(str(i["content"]).strip() for i in intention_items)

        tail_embeddings: list[list[float]] = []
        if tail_texts:
            tail_embeddings = await self.embed.embed_batch(tail_texts)

        emb_idx = 0
        summary_embedding: list[float] | None = None
        if summary:
            summary_embedding = tail_embeddings[emb_idx]
            emb_idx += 1

        intention_ids = await asyncio.to_thread(
            self._write_intentions_with_embeddings,
            intention_items,
            embeddings=tail_embeddings[emb_idx:],
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
        )
        stored_ids.extend(intention_ids)

        if summary and summary_embedding is not None:
            summary_node = MemoryNode(
                content=summary,
                layer=Layer.L3_SUMMARY,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                status=MemoryStatus.ACTIVE,
                is_latest=True,
                memory_at=memory_at,
            )
            summary_node.embedding = summary_embedding
            await asyncio.to_thread(self.vector.upsert, [summary_node])
            self.history.append(
                event="ADD",
                node_id=summary_node.node_id,
                user_id=user_id,
                old=None,
                new=summary_node.to_metadata(),
            )
            stored_ids.append(summary_node.node_id)

        return stored_ids, commit_result, False

    def _begin_summarize_task(
        self,
        *,
        content: str,
        current_time: str,
    ) -> asyncio.Task | None:
        """Start summarizer early so it can overlap extract / fast_write; cancel on reject."""
        if not self.settings.summarizer_enabled:
            return None
        if len(content) < self.summarizer.min_content_length:
            return None
        return asyncio.create_task(
            self.summarizer.summarize(content=content, current_time=current_time),
        )

    @staticmethod
    async def _cancel_summarize_task(
        task: asyncio.Task | None,
        *,
        reason: str = "",
    ) -> None:
        if task is None or task.done():
            return
        if reason:
            logger.info("summary cancelled after %s", reason)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    # ---- Internal helpers ------------------------------------------------------------

    @staticmethod
    def _normalize_memory_items(items: object) -> list[dict]:
        """Keep only structured items with non-blank content."""
        if not isinstance(items, list):
            return []
        normalized: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            normalized.append({**item, "content": content.strip()})
        return normalized

    @staticmethod
    def _canonical_content(content: str) -> str:
        text = unicodedata.normalize("NFKC", content).casefold()
        text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
        return " ".join(text.split())

    @classmethod
    def _dedup_memory_items(cls, items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        for item in items:
            key = cls._canonical_content(str(item.get("content", "")))
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _collect_new_memories(extracted: dict) -> tuple[list[str], list[dict]]:
        """Flatten extracted identity/fact items into parallel content and metadata lists."""
        texts: list[str] = []
        metas: list[dict] = []
        for item in extracted.get("identity") or []:
            content = item.get("content", "")
            if not content:
                continue
            texts.append(content)
            metas.append(
                {
                    "content": content,
                    "layer": "L4_IDENTITY",
                    "tags": item.get("tags") or [],
                    "speculate": item.get("speculate"),
                    "owner": item.get("owner", ""),
                }
            )
        for fact in extracted.get("facts") or []:
            content = fact.get("content", "")
            if not content:
                continue
            texts.append(content)
            metas.append(
                {
                    "content": content,
                    "layer": "L2_FACT",
                    "tags": fact.get("tags") or [],
                    "speculate": fact.get("speculate"),
                    "owner": fact.get("owner", ""),
                }
            )
        return texts, metas

    async def _fast_write(
        self,
        extracted: dict,
        metas: list[dict],
        *,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        memory_at: int | None,
        source_node_id: str = "",
    ) -> tuple[list[str], list[str]]:
        """Persist L0 (if any) + extracted L2/L4 in one embed_batch call.

        Returns ``(l0_ids, l2l4_ids)`` so the caller can enqueue only the L2/L4 nodes for
        reconcile without re-querying the store (L0 evolves via its own supersede chain and
        must never enter the reconcile queue).
        """
        l0_ids: list[str] = []
        l2l4_ids: list[str] = []
        prepared_l0 = None
        basic_info = extracted.get("basic_info")
        if isinstance(basic_info, dict) and basic_info:
            prepared_l0 = self.basic_profile_tool.prepare(
                arguments=basic_info,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )

        l2l4_nodes: list[MemoryNode] = []
        for meta in metas:
            layer_str = meta.get("layer") or "L2_FACT"
            try:
                layer = Layer(layer_str.upper())
            except ValueError:
                layer = Layer.L2_FACT
            l2l4_nodes.append(
                MemoryNode(
                    content=meta.get("content", ""),
                    layer=layer,
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    tags=list(meta.get("tags") or []),
                    status=MemoryStatus.ACTIVE,
                    is_latest=True,
                    speculate=meta.get("speculate"),
                    owner=meta.get("owner", ""),
                    memory_at=memory_at,
                    custom={"source_node_id": source_node_id} if source_node_id else None,
                )
            )

        texts: list[str] = []
        if prepared_l0 is not None:
            texts.append(prepared_l0.node.content)
        texts.extend(n.content for n in l2l4_nodes)

        if not texts:
            return l0_ids, l2l4_ids

        embeddings = await self.embed.embed_batch(texts)
        idx = 0
        if prepared_l0 is not None:
            l0_id = await asyncio.to_thread(
                self.basic_profile_tool.commit, prepared_l0, embeddings[idx]
            )
            l0_ids.append(l0_id)
            self.history.append(
                event="ADD",
                node_id=l0_id,
                user_id=user_id,
                old=None,
                new=prepared_l0.node.to_metadata(),
            )
            idx += 1

        kept_nodes: list[MemoryNode] = []
        kept_embeddings: list[list[float]] = []
        for node in l2l4_nodes:
            embedding = embeddings[idx]
            idx += 1
            if self.settings.vector_dedup_enabled and any(
                node.layer == previous.layer
                and self._cosine_similarity(embedding, previous_embedding)
                >= self.settings.vector_dedup_threshold
                for previous, previous_embedding in zip(
                    kept_nodes, kept_embeddings, strict=True
                )
            ):
                continue
            node.embedding = embedding
            kept_nodes.append(node)
            kept_embeddings.append(embedding)

        for node in kept_nodes:
            await asyncio.to_thread(self.vector.upsert, [node])
            l2l4_ids.append(node.node_id)
            self.history.append(
                event="ADD",
                node_id=node.node_id,
                user_id=user_id,
                old=None,
                new=node.to_metadata(),
            )

        logger.info(
            "fast_write app=%s user=%s n_nodes=%d (l0=%d l2l4=%d)",
            app_id,
            user_id,
            len(l0_ids) + len(l2l4_ids),
            len(l0_ids),
            len(l2l4_ids),
        )
        return l0_ids, l2l4_ids

    async def _apply_ops(
        self,
        ops: list[ReconcileOp],
        *,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        memory_at: int | None,
        source_node_id: str = "",
    ) -> list[str]:
        """Apply reconcile ops to the stores (soft-delete, add, supersede)."""
        stored_ids: list[str] = []
        for op in ops:
            if op.op == "SKIP":
                continue
            if op.op == "DELETE":
                old = (
                    await asyncio.to_thread(self.vector.get, op.memory_id)
                    if op.memory_id
                    else None
                )
                if old is None:
                    continue
                old_meta = old.to_metadata()
                old.status = MemoryStatus.SHADOW
                old.is_latest = False
                await asyncio.to_thread(self.vector.upsert, [old])
                self.history.append(
                    event="DELETE",
                    node_id=old.node_id,
                    user_id=user_id,
                    old=old_meta,
                    new=old.to_metadata(),
                )
                continue

            try:
                layer = Layer(op.layer.upper()) if op.layer else Layer.L2_FACT
            except ValueError:
                layer = Layer.L2_FACT
            node = MemoryNode(
                content=op.content or "",
                layer=layer,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                tags=list(op.tags),
                status=MemoryStatus.ACTIVE,
                is_latest=True,
                supersedes=list(op.supersedes),
                memory_at=memory_at,
                custom=build_apply_custom(
                    op,
                    vector=self.vector,
                    new_memory_at=memory_at,
                    extra=[memory_at] if memory_at is not None else None,
                    source_node_id=source_node_id,
                ) or None,
            )
            node.embedding = await self.embed.embed_queued(node.content)
            await asyncio.to_thread(self.vector.upsert, [node])
            stored_ids.append(node.node_id)
            self.history.append(
                event="SUPERSEDE" if op.supersedes else "ADD",
                node_id=node.node_id,
                user_id=user_id,
                old=None,
                new=node.to_metadata(),
            )

            for old_id in op.supersedes:
                await asyncio.to_thread(
                    self.vector.mark_superseded, old_id, superseded_by_id=node.node_id
                )

        return stored_ids

    def _write_intentions_with_embeddings(
        self,
        intentions: list[dict],
        *,
        embeddings: list[list[float]],
        app_id: str,
        user_id: str,
        agent_id: str,
    ) -> list[str]:
        """Persist L7 intention candidates into the graph (dual only); no-op otherwise."""
        graph = self.factory.graph
        if graph is None or not intentions:
            return []
        if len(embeddings) != len(intentions):
            logger.warning(
                "intention embed count mismatch: %d items vs %d vectors",
                len(intentions),
                len(embeddings),
            )
            return []

        ids: list[str] = []
        for intent, embedding in zip(intentions, embeddings, strict=True):
            content = intent.get("content", "").strip()
            node_id = str(uuid.uuid4())
            graph.add_node(
                GraphNode(
                    node_id=node_id,
                    layer=Layer.L7_INTENTION.value,
                    content=content,
                    app_id=app_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    embedding=embedding,
                    tags=intent.get("tags") or [],
                    gmt_created=int(time.time()),
                )
            )
            ids.append(node_id)
        return ids

