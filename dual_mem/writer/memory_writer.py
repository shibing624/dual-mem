from dataclasses import dataclass, field

from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryNode, MemoryStatus


@dataclass
class WriteResult:
    memory_id: str
    extra_node_ids: list[str] = field(default_factory=list)


class MemoryWriter:
    def __init__(self, *, factory: ComponentFactory, agent_mode: str):
        self.factory = factory
        self.agent_mode = agent_mode

    async def write(
        self,
        *,
        content: str,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        request_id: str,
        memory_at: int | None = None,
    ) -> WriteResult:
        node = MemoryNode(
            content=content,
            layer=Layer.L1_RAW,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            status=MemoryStatus.ACTIVE,
            memory_at=memory_at,
        )
        node.embedding = self.factory.embed.embed(content)
        self.factory.vector.upsert([node])
        self.factory.history.append(
            event="ADD",
            node_id=node.node_id,
            user_id=user_id,
            old=None,
            new=node.to_metadata(),
        )

        extra_node_ids: list[str] = []
        if self.agent_mode == "full":
            extra_node_ids = await self._run_cognition(raw=node, request_id=request_id)
            self.factory.vector.update_status(node.node_id, MemoryStatus.SHADOW)

        return WriteResult(memory_id=node.node_id, extra_node_ids=extra_node_ids)

    async def _run_cognition(self, *, raw: MemoryNode, request_id: str) -> list[str]:
        return []
