"""ultra 写侧：跑完 System1 后 fire-and-forget 入 System2 队列。

write 内部持有一个 MemoryWriter 跑 S1（与 pro 一致），完成后仅把 (user_id, app_id)
入队即返回，不阻塞写请求。System2 的实际加工由 run_system2_pending 在 digest 时消费。
"""

from dual_mem.registry import ComponentFactory
from dual_mem.system2.system2_agent import System2Agent
from dual_mem.writer.memory_writer import MemoryWriter, WriteResult


class System2Writer:
    def __init__(self, *, factory: ComponentFactory, agent_mode: str):
        self.factory = factory
        self.inner = MemoryWriter(factory=factory, agent_mode=agent_mode)
        self.processed_pairs: set[tuple[str, str]] = set()

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
        result = await self.inner.write(
            content=content,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            request_id=request_id,
            memory_at=memory_at,
        )
        self.factory.cache.enqueue_s2_task(user_id, app_id)
        return result

    async def run_system2_pending(self) -> int:
        agent = System2Agent(factory=self.factory)
        self.processed_pairs = set()
        processed = 0
        while True:
            task = self.factory.cache.dequeue_s2_task()
            if task is None:
                break
            agent.run(app_id=task["app_id"], user_id=task["user_id"])
            self.processed_pairs.add((task["app_id"], task["user_id"]))
            processed += 1
        return processed
