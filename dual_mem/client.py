import json
import time
import uuid

from dual_mem.config import Settings
from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.reader import Reader
from dual_mem.system2.system2_writer import System2Writer
from dual_mem.types import MemoryStatus
from dual_mem.writer.memory_writer import MemoryWriter


class MemoryClient:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        storage_dir: str | None = None,
        mode: str | None = None,
        embed=None,
        llm=None,
    ):
        if settings is None:
            overrides = {}
            if storage_dir is not None:
                overrides["storage_dir"] = storage_dir
            if mode is not None:
                overrides["mode"] = mode
            settings = Settings(**overrides)
        self.settings = settings
        self.mode = mode or settings.mode

        factory_kwargs = {"settings": settings}
        if embed is not None:
            factory_kwargs["embed"] = embed
        if llm is not None:
            factory_kwargs["llm"] = llm
        self.factory = ComponentFactory(**factory_kwargs)

        if self.settings.mode == "ultra":
            self.writer = System2Writer(
                factory=self.factory, agent_mode=self.settings.agent_mode
            )
        else:
            self.writer = MemoryWriter(
                factory=self.factory, agent_mode=self.settings.agent_mode
            )
        self.reader = Reader(factory=self.factory)

    async def add(
        self,
        *,
        content: str = "",
        messages: list | None = None,
        app_id: str,
        user_id: str,
        agent_id: str = "",
        session_id: str = "",
        memory_at: int | None = None,
        mode: str | None = None,
    ) -> dict:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        text = content if content else json.dumps(messages, ensure_ascii=False)
        result = await self.writer.write(
            content=text,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            request_id=request_id,
            memory_at=memory_at,
        )
        return {
            "success": True,
            "memory_id": result.memory_id,
            "request_id": request_id,
            "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    async def search(
        self,
        *,
        query: str,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None = None,
        session_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
        created_after: int | None = None,
    ) -> dict:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        memories = self.reader.search(
            query=query,
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            limit=limit,
            min_score=min_score,
            profile_limit=profile_limit,
            profile_min_score=profile_min_score,
            created_after=created_after,
        )
        return {
            "success": True,
            "request_id": request_id,
            "memories": memories,
            "processing_time_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    async def get(self, memory_id: str) -> dict | None:
        node = self.factory.vector.get(memory_id)
        if node is None:
            return None
        return Reader._to_dict(node)

    async def list(
        self, *, app_id: str, user_id: str, agent_id: str = "", limit: int = 100
    ) -> list[dict]:
        where = build_filter(
            app_ids=[app_id],
            user_id=user_id,
            agent_ids=[agent_id],
            statuses=[MemoryStatus.ACTIVE],
        )
        nodes = self.factory.vector.get_many(where, limit=limit)
        return [Reader._to_dict(node) for node in nodes]

    async def update(self, memory_id: str, content: str) -> dict:
        node = self.factory.vector.get(memory_id)
        if node is None:
            return {"success": False, "error_code": 404}
        old_meta = node.to_metadata()
        node.content = content
        node.embedding = self.factory.embed.embed(content)
        node.gmt_modified = int(time.time())
        self.factory.vector.upsert([node])
        self.factory.history.append(
            event="UPDATE",
            node_id=node.node_id,
            user_id=node.user_id,
            old=old_meta,
            new=node.to_metadata(),
        )
        return {"success": True, "memory_id": memory_id}

    async def delete(self, memory_id: str) -> dict:
        node = self.factory.vector.get(memory_id)
        if node is None:
            return {"success": False, "error_code": 404}
        self.factory.vector.delete([memory_id])
        self.factory.history.append(
            event="DELETE",
            node_id=memory_id,
            user_id=node.user_id,
            old=node.to_metadata(),
            new=None,
        )
        return {"success": True}

    async def delete_bulk(
        self,
        *,
        app_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        confirm: bool = False,
    ) -> dict:
        if confirm is not True:
            return {"success": False, "error_code": 400}
        where: dict = {"app_id": {"$in": [app_id]}}
        if user_id is not None:
            where["user_id"] = user_id
        if agent_id is not None:
            where["agent_id"] = agent_id
        nodes = self.factory.vector.get_many(where)
        node_ids = [node.node_id for node in nodes]
        self.factory.vector.delete(node_ids)
        for node in nodes:
            self.factory.history.append(
                event="DELETE",
                node_id=node.node_id,
                user_id=node.user_id,
                old=node.to_metadata(),
                new=None,
            )
        return {"success": True, "deleted": len(node_ids)}

    async def digest(self) -> dict:
        if isinstance(self.writer, System2Writer):
            processed = await self.writer.run_system2_pending()
            return {"success": True, "processed": processed}
        return {"success": True, "processed": 0}
