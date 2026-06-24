# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MemAgent orchestrating the System1 cognitive pipeline. Defaults to fast-write
(extract -> direct ADD into L2/L4, queue reconcile for the System2 background worker) so the
write path stays light. When reconcile_sync=True we run reconcile inline for strong-consistency
contracts. Also handles L7 intentions (dual graph), L0 basic profile, and L3 summary.
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime

from dual_mem.agent.basic_profile import BasicProfileTool
from dual_mem.agent.extractor import Extractor
from dual_mem.agent.gate import AttentionalGate, GateConfig
from dual_mem.agent.reconciler import ReconcileOp, Reconciler
from dual_mem.agent.summarizer import Summarizer
from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.sdk_models import GateResult
from dual_mem.storage.graph_store import GraphNode
from dual_mem.types import Layer, MemoryNode, MemoryStatus

logger = logging.getLogger("dual_mem.agent.mem")


class MemAgent:
    """Coordinates gate, extractor, reconciler and summarizer for one input memory."""

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
        self.extractor = Extractor(
            llm=llm,
            max_content_chars=self.settings.extract_max_content_chars,
            retry_on_failure=self.settings.extract_retry_on_failure,
        )
        self.summarizer = Summarizer(
            llm=llm,
            min_content_length=self.settings.summarizer_min_content_length,
        )
        self.reconciler = Reconciler(
            llm=llm,
            embed=self.embed,
            vector=self.vector,
            enable_search_query=self.settings.reconcile_search_query,
        )
        self.gate = AttentionalGate(
            threshold=self.settings.gate_threshold,
            llm=llm,
            config=GateConfig(
                threshold=self.settings.gate_threshold,
                heuristic_shortcircuit=self.settings.gate_heuristic_shortcircuit,
                shortcircuit_novelty=self.settings.gate_shortcircuit_novelty,
                shortcircuit_relevance=self.settings.gate_shortcircuit_relevance,
                shortcircuit_reject_novelty=self.settings.gate_shortcircuit_reject_novelty,
            ),
        )

    async def run(
        self,
        *,
        raw_node: MemoryNode,
        content: str,
        embedding: list[float],
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        request_id: str,
        memory_at: int | None,
        user_queries: list[str] | None = None,
        agent_context: str | None = None,
        gate_turn_embeddings: list[list[float]] | None = None,
    ) -> tuple[list[str], GateResult, bool]:
        """Run the System1 pipeline for one raw memory.

        Returns (extra_node_ids, gate_result, is_ephemeral). Caller decides what to do with
        the raw L1 node based on extra_node_ids being non-empty (shadow it).

        ``user_queries`` carries the per-turn user-only texts for multi-turn writes; when
        present each turn is embedded separately so Gate novelty = max(1 - max_sim) across
        turns. None means single-turn fallback to whole-content novelty.
        """
        current_time = (
            datetime.fromtimestamp(memory_at).isoformat(timespec="seconds") if memory_at else ""
        )

        use_combined = self.settings.combined_gate_extract and self.settings.gate_enabled
        summary_task: asyncio.Task | None = None

        if use_combined:
            sims = await self._gate_similarity_context(
                embedding=embedding,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                user_queries=user_queries,
                gate_turn_embeddings=gate_turn_embeddings,
            )
            shortcircuited = await self.gate.try_shortcircuit_pass(
                content=content,
                existing_similarities=sims,
            )
            if shortcircuited is None:
                rejected = await self.gate.try_shortcircuit_reject(
                    content=content,
                    existing_similarities=sims,
                )
                if rejected is not None:
                    logger.info(
                        "gate REJECT (short-circuit) score=%.3f novelty=%.3f reason=%s",
                        rejected.gate_score, rejected.novelty, rejected.reason,
                    )
                    return [], rejected, False
            include_gate = shortcircuited is None
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
                include_gate=include_gate,
            )
            if shortcircuited is not None:
                gate_result = shortcircuited
                if not (
                    extracted.get("identity")
                    or extracted.get("facts")
                    or extracted.get("intentions")
                ):
                    logger.warning(
                        "gate short-circuit PASS but extract empty (content len=%d)",
                        len(content),
                    )
            else:
                gate_result = await self.gate.finalize_from_llm(
                    content=content,
                    llm_scores=extracted.get("gate_decision"),
                    existing_similarities=sims,
                    scoring_method="llm",
                )
        else:
            gate_result = await self._evaluate_gate(
                content=content,
                embedding=embedding,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                user_queries=user_queries,
                agent_context=agent_context,
                gate_turn_embeddings=gate_turn_embeddings,
            )
            if self.settings.gate_enabled and not gate_result.passed:
                return [], gate_result, False

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
            )

        try:
            self.cache.log_pipeline(
                request_id=request_id,
                stage="GATE",
                payload={
                    "passed": gate_result.passed,
                    "score": gate_result.gate_score,
                    "novelty": gate_result.novelty,
                    "reason": gate_result.reason,
                    "combined": use_combined,
                },
            )
        except Exception:
            pass
        logger.info(
            "gate %s score=%.3f novelty=%.3f reason=%s",
            "PASS" if gate_result.passed else "REJECT",
            gate_result.gate_score, gate_result.novelty, gate_result.reason,
        )
        if self.settings.gate_enabled and not gate_result.passed:
            await self._cancel_summarize_task(summary_task, reason="gate reject")
            return [], gate_result, False

        try:
            self.cache.log_pipeline(
                request_id=request_id,
                stage="EXTRACT",
                payload={
                    "n_identity": len(extracted.get("identity") or []),
                    "n_facts": len(extracted.get("facts") or []),
                    "n_intentions": len(extracted.get("intentions") or []),
                    "is_ephemeral": bool(extracted.get("is_ephemeral")),
                },
            )
        except Exception:
            pass
        logger.info(
            "extract identity=%d facts=%d intentions=%d ephemeral=%s",
            len(extracted.get("identity") or []),
            len(extracted.get("facts") or []),
            len(extracted.get("intentions") or []),
            bool(extracted.get("is_ephemeral")),
        )
        if extracted.get("is_ephemeral"):
            await self._cancel_summarize_task(summary_task, reason="extract ephemeral")
            return [], gate_result, True

        emotion = extracted.get("emotion") or {}
        new_memories, new_meta = self._collect_new_memories(extracted)
        basic_info_present = isinstance(extracted.get("basic_info"), dict) and bool(
            extracted["basic_info"]
        )

        stored_ids: list[str] = []

        # ---- Step 3: Persist L0/L2/L4 nodes ---------------------------------------------
        # reconcile_sync writes L0/L2/L4 first then reconciles inline (strong consistency);
        # the default path fast-writes and hands only the L2/L4 nodes to the async reconcile
        # worker (L0 evolves via its own supersede chain and never enters that queue).
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
                    emotion=emotion,
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
                        emotion=emotion,
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
                    emotion=emotion,
                )
                stored_ids = l0_ids + l2l4_ids
                if l2l4_ids:
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
                pass

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

        intention_ids = self._write_intentions_with_embeddings(
            intention_items,
            embeddings=tail_embeddings[emb_idx:],
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            emotion=emotion,
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
            self.vector.upsert([summary_node])
            self.history.append(
                event="ADD",
                node_id=summary_node.node_id,
                user_id=user_id,
                old=None,
                new=summary_node.to_metadata(),
            )
            stored_ids.append(summary_node.node_id)

        return stored_ids, gate_result, False

    def _begin_summarize_task(
        self,
        *,
        content: str,
        current_time: str,
    ) -> asyncio.Task | None:
        """Start summarizer early so it can overlap extract / fast_write; cancel on reject."""
        if not self.settings.summarizer_enabled:
            return None
        if len(content) < self.settings.summarizer_min_content_length:
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

    async def _maybe_summarize(self, *, content: str, current_time: str) -> str | None:
        if not self.settings.summarizer_enabled:
            return None
        return await self.summarizer.summarize(content=content, current_time=current_time)

    # ---- Internal helpers ------------------------------------------------------------

    async def _evaluate_gate(
        self,
        *,
        content: str,
        embedding: list[float],
        app_id: str,
        user_id: str,
        agent_id: str,
        user_queries: list[str] | None = None,
        agent_context: str | None = None,
        gate_turn_embeddings: list[list[float]] | None = None,
    ) -> GateResult:
        """Evaluate the attentional gate using existing L2/L4 hits as similarity context."""
        if not self.settings.gate_enabled:
            return GateResult(
                passed=True,
                gate_score=1.0,
                novelty=1.0,
                biographical_relevance=0.0,
                emotional_arousal=0.0,
                reason="gate disabled",
                scoring_method="bypass",
            )

        sims = await self._gate_similarity_context(
            embedding=embedding,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            user_queries=user_queries,
            gate_turn_embeddings=gate_turn_embeddings,
        )
        if isinstance(sims, list) and sims and isinstance(sims[0], list):
            per_turn_sims: list[list[dict]] = sims  # type: ignore[assignment]
            return await self.gate.evaluate(
                content=content,
                existing_similarities=per_turn_sims,
                agent_context=agent_context,
            )
        return await self.gate.evaluate(
            content=content,
            existing_similarities=sims,  # type: ignore[arg-type]
            agent_context=agent_context,
        )

    async def _gate_similarity_context(
        self,
        *,
        embedding: list[float],
        app_id: str,
        user_id: str,
        agent_id: str,
        user_queries: list[str] | None = None,
        gate_turn_embeddings: list[list[float]] | None = None,
    ) -> list[dict] | list[list[dict]]:
        """Vector similarity hits for gate novelty (multi-turn or single-turn)."""
        where = build_filter(
            app_ids=[app_id],
            user_id=user_id,
            agent_ids=[agent_id],
            layers=[Layer.L2_FACT, Layer.L4_IDENTITY],
            statuses=[MemoryStatus.ACTIVE],
        )

        if user_queries:
            if gate_turn_embeddings is not None:
                turn_embs = gate_turn_embeddings
            else:
                try:
                    turn_embs = await self.embed.embed_batch(user_queries)
                except Exception:
                    turn_embs = [embedding]

            async def _hits_for_emb(emb: list[float]) -> list[dict]:
                try:
                    nodes = await asyncio.to_thread(
                        self.vector.query,
                        embedding=emb,
                        where=where,
                        top_k=5,
                    )
                    return [
                        {"node_id": h.node_id, "score": h.score}
                        for h in nodes
                        if h.score >= 0.3
                    ]
                except Exception:
                    return []

            per_turn = await asyncio.gather(*[_hits_for_emb(e) for e in turn_embs])
            return list(per_turn)

        try:
            nodes = await asyncio.to_thread(
                self.vector.query,
                embedding=embedding,
                where=where,
                top_k=5,
            )
            return [
                {"node_id": h.node_id, "score": h.score}
                for h in nodes
                if h.score >= 0.3
            ]
        except Exception:
            return []

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
        emotion: dict,
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
                    memory_at=memory_at,
                    custom=_emotion_custom(emotion) or None,
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
            l0_id = self.basic_profile_tool.commit(prepared_l0, embeddings[idx])
            l0_ids.append(l0_id)
            self.history.append(
                event="ADD",
                node_id=l0_id,
                user_id=user_id,
                old=None,
                new=prepared_l0.node.to_metadata(),
            )
            idx += 1

        for node in l2l4_nodes:
            node.embedding = embeddings[idx]
            idx += 1
            self.vector.upsert([node])
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
        emotion: dict,
    ) -> list[str]:
        """Apply reconcile ops to the stores (soft-delete, add, supersede)."""
        stored_ids: list[str] = []
        for op in ops:
            if op.op == "DELETE":
                old = self.vector.get(op.memory_id) if op.memory_id else None
                if old is None:
                    continue
                old_meta = old.to_metadata()
                old.status = MemoryStatus.SHADOW
                old.is_latest = False
                self.vector.upsert([old])
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
                custom=_merge_custom(_emotion_custom(emotion), _reflect_custom(op)) or None,
            )
            node.embedding = await self.embed.embed_queued(node.content)
            self.vector.upsert([node])
            stored_ids.append(node.node_id)
            self.history.append(
                event="SUPERSEDE" if op.supersedes else "ADD",
                node_id=node.node_id,
                user_id=user_id,
                old=None,
                new=node.to_metadata(),
            )

            for old_id in op.supersedes:
                self.vector.mark_superseded(old_id, superseded_by_id=node.node_id)

        return stored_ids

    def _write_intentions_with_embeddings(
        self,
        intentions: list[dict],
        *,
        embeddings: list[list[float]],
        app_id: str,
        user_id: str,
        agent_id: str,
        emotion: dict,
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
                    custom=_emotion_custom(emotion) or None,
                )
            )
            ids.append(node_id)
        return ids


def _emotion_custom(emotion: dict) -> dict:
    """Pack non-zero emotion fields into a custom dict for storage on a memory node."""
    out: dict = {}
    if not isinstance(emotion, dict):
        return out
    valence = float(emotion.get("valence", 0.0) or 0.0)
    arousal = float(emotion.get("arousal", 0.0) or 0.0)
    dominant = emotion.get("dominant_emotion")
    if abs(valence) > 1e-6:
        out["emotional_valence"] = valence
    if arousal > 1e-6:
        out["emotional_arousal"] = arousal
    if isinstance(dominant, str) and dominant:
        out["dominant_emotion"] = dominant
    return out


def _reflect_custom(op: ReconcileOp) -> dict:
    """Pack reflector metadata (update_type, temporal_scope, negation) into node.custom."""
    out: dict = {}
    if op.update_type:
        out["update_type"] = op.update_type
    if op.temporal_scope:
        out["temporal_scope"] = op.temporal_scope
    if op.negation:
        out["negation"] = True
    return out


def _merge_custom(*parts: dict) -> dict:
    """Shallow-merge custom dicts; later parts override earlier on key clash."""
    out: dict = {}
    for part in parts:
        if part:
            out.update(part)
    return out
