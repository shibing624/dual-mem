# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Shared async memory operations for REST, Python MCP, and HTTP-based clients
(e.g. future npm MCP). Each ``memory_*`` method mirrors an MCP tool and returns plain
JSON-serializable dicts/lists via sdk_models ``.to_dict()``.
"""
from dual_mem.client import MemoryClient


class MemoryOperations:
    """Thin wrapper over MemoryClient; single source of truth for tool semantics."""

    def __init__(self, client: MemoryClient) -> None:
        self.client = client

    async def memory_add(
        self,
        *,
        user_id: str,
        app_id: str | None = None,
        content: str = "",
        messages: list[dict] | None = None,
        agent_id: str = "",
        session_id: str = "",
        memory_at: int | None = None,
    ) -> dict:
        result = await self.client.add(
            content=content,
            messages=messages,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            memory_at=memory_at,
        )
        return result.to_dict()

    async def memory_search(
        self,
        *,
        query: str,
        user_id: str,
        app_ids: list[str] | None = None,
        agent_ids: list[str] | None = None,
        session_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        include_derived: bool = True,
        created_after: int | None = None,
        debug: bool = False,
    ) -> dict:
        result = await self.client.search(
            query=query,
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            limit=limit,
            min_score=min_score,
            profile_limit=profile_limit,
            profile_min_score=profile_min_score,
            intention_limit=intention_limit,
            include_derived=include_derived,
            created_after=created_after,
            debug=debug,
        )
        return result.to_dict()

    async def memory_list(
        self,
        *,
        user_id: str,
        app_id: str | None = None,
        agent_id: str = "",
        limit: int = 100,
    ) -> list[dict]:
        items = await self.client.list(
            app_id=app_id, user_id=user_id, agent_id=agent_id, limit=limit
        )
        return [item.to_dict() for item in items]

    async def memory_get(self, memory_id: str) -> dict | None:
        item = await self.client.get(memory_id)
        return item.to_dict() if item is not None else None

    async def memory_update(self, memory_id: str, content: str) -> dict:
        result = await self.client.update(memory_id, content)
        return result.to_dict()

    async def memory_delete(self, memory_id: str) -> dict:
        result = await self.client.delete(memory_id)
        return result.to_dict()

    async def memory_delete_scope(
        self,
        *,
        confirm: bool,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict:
        result = await self.client.delete_bulk(
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            confirm=confirm,
        )
        return result.to_dict()

    async def memory_list_scopes(
        self,
        *,
        app_id: str | None = None,
        limit: int = 5000,
    ) -> list[dict]:
        scopes = await self.client.list_scopes(app_id=app_id, limit=limit)
        return [s.to_dict() for s in scopes]

    async def memory_digest(self) -> dict:
        result = await self.client.digest()
        return result.to_dict()

    async def aclose(self) -> None:
        await self.client.aclose()
