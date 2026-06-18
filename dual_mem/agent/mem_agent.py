"""MemAgent：System1 同步认知层编排。

extract（含 L0 工具）→ collect new memories → reconcile → 应用 ops（ADD/EVOLVE/DELETE）
→ summarize（长内容产 L3）。返回本次认知层写入的所有 node_id。
"""

from datetime import datetime

from dual_mem.agent.basic_profile import BasicProfileTool
from dual_mem.agent.extractor import Extractor
from dual_mem.agent.reconciler import ReconcileOp, Reconciler
from dual_mem.agent.summarizer import Summarizer
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryNode, MemoryStatus


class MemAgent:
    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory
        self.vector = factory.vector
        self.embed = factory.embed
        self.history = factory.history
        self.basic_profile_tool = BasicProfileTool(vector=self.vector, embed=self.embed)
        self.extractor = Extractor(llm=factory.llm, basic_profile_tool=self.basic_profile_tool)
        self.summarizer = Summarizer(llm=factory.llm)
        self.reconciler = Reconciler(llm=factory.llm, embed=self.embed, vector=self.vector)

    def run(
        self,
        *,
        raw_node: MemoryNode,
        content: str,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        request_id: str,
        memory_at: int | None,
    ) -> list[str]:
        current_time = (
            datetime.fromtimestamp(memory_at).isoformat(timespec="seconds") if memory_at else ""
        )

        extracted = self.extractor.extract(
            content=content,
            current_time=current_time,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

        new_memories, new_memories_meta = self._collect_new_memories(extracted)

        stored_ids: list[str] = []
        if new_memories:
            ops = self.reconciler.reconcile(
                new_memories=new_memories,
                new_memories_meta=new_memories_meta,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                current_time=current_time,
            )
            stored_ids = self._apply_ops(
                ops,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                memory_at=memory_at,
            )

        summary = self.summarizer.summarize(content=content, current_time=current_time)
        if summary:
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
            summary_node.embedding = self.embed.embed(summary)
            self.vector.upsert([summary_node])
            self.history.append(
                event="ADD",
                node_id=summary_node.node_id,
                user_id=user_id,
                old=None,
                new=summary_node.to_metadata(),
            )
            stored_ids.append(summary_node.node_id)

        return stored_ids

    @staticmethod
    def _collect_new_memories(extracted: dict) -> tuple[list[str], list[dict]]:
        texts: list[str] = []
        metas: list[dict] = []
        for item in extracted.get("identity") or []:
            content = item.get("content", "")
            if not content:
                continue
            texts.append(content)
            metas.append({"content": content, "layer": "L4_IDENTITY", "tags": item.get("tags") or []})
        for fact in extracted.get("facts") or []:
            content = fact.get("content", "")
            if not content:
                continue
            texts.append(content)
            metas.append({"content": content, "layer": "L2_FACT", "tags": fact.get("tags") or []})
        return texts, metas

    def _apply_ops(
        self,
        ops: list[ReconcileOp],
        *,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        memory_at: int | None,
    ) -> list[str]:
        stored_ids: list[str] = []
        for op in ops:
            if op.op == "DELETE":
                old = self.vector.get(op.memory_id)
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

            layer = Layer(op.layer.upper()) if op.layer else Layer.L2_FACT
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
            )
            node.embedding = self.embed.embed(node.content)
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
                old = self.vector.get(old_id)
                if old is None:
                    continue
                old.is_latest = False
                if node.node_id not in old.superseded_by:
                    old.superseded_by.append(node.node_id)
                old.status = MemoryStatus.SUPERSEDED
                self.vector.upsert([old])

        return stored_ids
