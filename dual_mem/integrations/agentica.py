# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: agentica 集成层。

agentica 的 ``Agent`` 通过 ``workspace`` 对象消费长期记忆，核心协议方法：
- ``get_context_prompt()`` / ``get_frozen_context()`` : 文件型上下文（AGENTS.md /
  PERSONA.md / USER.md 等）
- ``get_relevant_memories(query, limit, already_surfaced)`` : 注入 system prompt 的
  事实召回
- ``get_relevant_experiences(query, limit)`` : 经验召回
- ``save_memory(content)`` / ``write_memory_entry(...)`` : 记忆写入（被内置
  ``BuiltinMemoryTool`` 调用）

本模块提供两层接入：

1. ``DualMemMemory`` —— 继承 ``MemoryBackend`` 的 agentica 记忆模块，暴露
   ``add_messages`` / ``retrieve``（对齐 docs/skill 约定），并可经 ``as_workspace()``
   生成直接喂给 ``Agent`` 的 workspace。
2. ``DualMemWorkspace`` —— 组合一个原生 agentica ``Workspace``（承接文件型上下文 /
   skills / USER.md），仅把「语义记忆」相关方法覆盖为调用 dual_mem ``MemoryClient``。
   于是直接 ``Agent(workspace=DualMemWorkspace(...))`` 即可，无需改动 agentica 框架代码。

agentica 是可选依赖：本模块在 import 时不强制要求它已安装，仅在真正构建
``DualMemWorkspace`` 时才懒加载。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from dual_mem.integrations._base import (
    MemoryBackend,
    _SKIP_QUERIES,
)

logger = logging.getLogger("dual_mem.integrations.agentica")


def _ensure_agentica():
    """懒加载 agentica 的可选依赖；缺失时给出清晰的报错。"""
    try:
        from agentica.memory.models import MemoryEntry
        from agentica.workspace import Workspace

        return Workspace, MemoryEntry
    except Exception as exc:  # agentica 未安装
        raise RuntimeError(
            "agentica is required for the Workspace adapter. "
            "Install it with `pip install agentica` or `pip install 'dual-mem[agentica]'`."
        ) from exc


class DualMemMemory(MemoryBackend):
    """agentica 长期记忆模块：基于 dual_mem 的 MemoryClient。

    两种用法：

    1) 手动把召回结果注入 agentica Agent 的 system prompt::

           mem = get_backend("agentica", user_id="u1")
           ctx = await mem.retrieve("用户偏好什么咖啡？", user_id="u1")
           await agent.aprint_response("...", context=ctx)

    2) 作为 Agent 的原生长期记忆（自动注入 + 自动保存）::

           ws = mem.as_workspace(workspace_dir=".agentica")
           agent = Agent(workspace=ws, enable_long_term_memory=True)
    """

    def __init__(
        self,
        *,
        user_id: str = "",
        agent_id: str = "",
        app_id: Optional[str] = None,
        settings: Optional[Any] = None,
        client: Optional[Any] = None,
        storage_dir: Optional[str] = None,
        mode: Optional[str] = None,
        embed: Optional[Any] = None,
        llm: Optional[Any] = None,
    ) -> None:
        super().__init__(
            settings=settings,
            client=client,
            storage_dir=storage_dir,
            mode=mode,
            embed=embed,
            llm=llm,
        )
        self._user_id = user_id
        self._agent_id = agent_id
        self._app_id = app_id

    async def add_messages(
        self,
        messages: list,
        *,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: str = "",
        memory_at: Optional[int] = None,
    ) -> Any:
        """把一轮多轮对话存为记忆（支持 agentica 的 ChatMessage / dict 列表）。"""
        uid = user_id or self._user_id
        if not uid:
            raise ValueError("user_id is required (pass explicitly or set on the backend)")
        return await self.add(
            content="",
            messages=messages,
            user_id=uid,
            app_id=app_id or self._app_id,
            agent_id=agent_id or self._agent_id,
            session_id=session_id,
            memory_at=memory_at,
        )

    async def retrieve(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        max_chars: int = 2000,
    ) -> str:
        """检索并把相关记忆格式化为 <relevant-memories> 注入块。"""
        uid = user_id or self._user_id
        if not uid:
            raise ValueError("user_id is required (pass explicitly or set on the backend)")
        return await self.render_context(
            query=query,
            user_id=uid,
            limit=limit,
            min_score=min_score,
            profile_limit=profile_limit,
            profile_min_score=profile_min_score,
            intention_limit=intention_limit,
            max_chars=max_chars,
        )

    def as_workspace(
        self,
        *,
        workspace_dir: Optional[str] = None,
        config: Optional[Any] = None,
        agent_id: Optional[str] = None,
        app_id: Optional[str] = None,
        max_recall: int = 5,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        use_experiences: bool = False,
        frozen_memory: Optional[str] = None,
    ) -> "DualMemWorkspace":
        """生成可直接传给 ``Agent(workspace=...)`` 的 Workspace 适配器。"""
        return DualMemWorkspace(
            self,
            workspace_dir=workspace_dir,
            config=config,
            user_id=self._user_id,
            agent_id=agent_id or self._agent_id,
            app_id=app_id or self._app_id,
            max_recall=max_recall,
            min_score=min_score,
            profile_limit=profile_limit,
            profile_min_score=profile_min_score,
            intention_limit=intention_limit,
            use_experiences=use_experiences,
            frozen_memory=frozen_memory,
        )


class DualMemWorkspace:
    """agentica ``Workspace`` 协议适配器：用 dual_mem 承接语义长期记忆。

    组合一个原生 agentica ``Workspace``（负责 AGENTS.md / PERSONA.md / USER.md /
    skills 等文件型上下文），仅重写与「语义记忆」相关的几个方法，使其调用 dual_mem
    的 ``MemoryClient``。对 ``Agent`` 而言它与原生 ``Workspace`` 完全等价，因此
    ``Agent(workspace=DualMemWorkspace(...))`` 即可启用 dual_mem 作为长期记忆后端。

    内置的 ``BuiltinMemoryTool``（``save_memory`` / ``search_memory``）也会被一并接管：
    - ``save_memory`` / ``write_memory_entry`` → ``MemoryClient.add``
    - ``search_memory`` → 返回最近一次 system-prompt 召回的 dual_mem 结果
    """

    def __init__(
        self,
        backend_or_client: Any,
        *,
        workspace_dir: Optional[str] = None,
        config: Optional[Any] = None,
        user_id: Optional[str] = None,
        agent_id: str = "",
        app_id: Optional[str] = None,
        max_recall: int = 5,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
        intention_limit: int = 0,
        use_experiences: bool = False,
        frozen_memory: Optional[str] = None,
    ) -> None:
        Workspace, _ = _ensure_agentica()
        if isinstance(backend_or_client, MemoryBackend):
            self._backend = backend_or_client
        else:
            self._backend = MemoryBackend(client=backend_or_client)

        self._user_id = user_id
        self._agent_id = agent_id
        self._app_id = app_id
        self._max_recall = max_recall
        self._min_score = min_score
        self._profile_limit = profile_limit
        self._profile_min_score = profile_min_score
        self._intention_limit = intention_limit
        self._use_experiences = use_experiences
        self._frozen_memory = frozen_memory
        self._last_recall: list = []

        # 文件型上下文 / skills / USER.md 仍由原生 Workspace 负责
        self._ws = Workspace(
            path=workspace_dir,
            user_id=user_id,
            config=config,
        )

    # ---- 身份 / 路径（转发或直读） -----------------------------------------
    @property
    def user_id(self) -> Optional[str]:
        return self._user_id or self._ws.user_id

    @property
    def path(self):
        return self._ws.path

    # ---- 语义记忆：覆盖原生 Workspace 方法 ---------------------------------
    async def get_relevant_memories(
        self, query: str = "", limit: int = 5, already_surfaced: Optional[list] = None
    ) -> str:
        if not query or query.strip().lower() in _SKIP_QUERIES:
            return ""
        surfaced: set[str] = set()
        if already_surfaced:
            for item in already_surfaced:
                if isinstance(item, (tuple, list)) and len(item) > 1:
                    surfaced.add((item[1] or "").strip())

        items = await self._search(query, limit)
        kept = [m for m in items if (m.content or "").strip() not in surfaced]
        self._last_recall = self._build_entries(kept)
        return self._format_recall(kept, query, header="## Memory")

    async def get_relevant_experiences(self, query: str = "", limit: int = 5) -> str:
        if not self._use_experiences or not query:
            return ""
        items = await self._search(query, limit)
        self._last_recall = self._build_entries(items)
        return self._format_recall(items, query, header="## Experience")

    async def save_memory(self, content: str, long_term: bool = False) -> Any:
        return await self._backend.add(
            content=content,
            user_id=self.user_id,
            app_id=self._app_id,
            agent_id=self._agent_id,
        )

    async def write_memory_entry(
        self,
        title: str,
        content: str,
        memory_type: str = "project",
        description: str = "",
        sync_to_global_agent_md: bool = False,
    ) -> str:
        res = await self._backend.add(
            content=content,
            user_id=self.user_id,
            app_id=self._app_id,
            agent_id=self._agent_id,
        )
        memory_id = getattr(res, "memory_id", None) if res is not None else None
        return memory_id or f"dual-mem://{title}"

    def get_frozen_memory(self) -> Optional[str]:
        return self._frozen_memory

    def set_frozen_memory(self, text: Optional[str]) -> None:
        self._frozen_memory = text

    # ---- 让内置 search_memory 工具也能返回 dual_mem 结果 -------------------
    def _collect_searchable_memory_entries(self) -> list:
        return self._last_recall

    def compute_relevance_score(self, query: str, content: str) -> float:
        # 最近一次召回已由 dual_mem 完成语义排序；此处直接放行，交给工具排序。
        return 1.0

    # ---- 内部辅助 ----------------------------------------------------------
    async def _search(self, query: str, limit: int) -> list:
        result = await self._backend.search(
            query=query,
            user_id=self.user_id,
            app_ids=[self._app_id] if self._app_id else None,
            agent_ids=[self._agent_id] if self._agent_id else None,
            limit=limit,
            min_score=self._min_score,
            profile_limit=self._profile_limit,
            profile_min_score=self._profile_min_score,
            intention_limit=self._intention_limit,
        )
        return result.memories.flatten(limit=limit)

    def _build_entries(self, items: list) -> list:
        _, MemoryEntry = _ensure_agentica()
        entries = []
        for m in items:
            mid = getattr(m, "memory_id", None) or ""
            entries.append(
                MemoryEntry(
                    name=mid or getattr(m, "category", ""),
                    description=(m.content or "")[:80],
                    content=m.content or "",
                    memory_type="project",
                    file_path=f"dual-mem://{mid}" if mid else "",
                )
            )
        return entries

    @staticmethod
    def _format_recall(items: list, query: str, *, header: str) -> str:
        if not items:
            return ""
        bullets = "\n".join(f"- {(m.content or '').strip()}" for m in items)
        return f"Relevant memories for {query}:\n{header}\n{bullets}"

    # ---- 其余 Workspace 协议方法转发给原生 Workspace -----------------------
    def __getattr__(self, name: str) -> Any:
        ws = self.__dict__.get("_ws")
        if ws is None:
            raise AttributeError(name)
        return getattr(ws, name)


def get_agentica_memory_backend(
    *,
    client: Optional[Any] = None,
    settings: Optional[Any] = None,
    storage_dir: Optional[str] = None,
    mode: Optional[str] = None,
    embed: Optional[Any] = None,
    llm: Optional[Any] = None,
    workspace_dir: Optional[str] = None,
    config: Optional[Any] = None,
    user_id: Optional[str] = None,
    agent_id: str = "",
    app_id: Optional[str] = None,
    max_recall: int = 5,
    min_score: float = 0.4,
    profile_limit: int = -1,
    profile_min_score: float = 0.3,
    intention_limit: int = 0,
    use_experiences: bool = False,
    frozen_memory: Optional[str] = None,
) -> DualMemWorkspace:
    """构建可直接传给 ``Agent(workspace=...)`` 的 dual_mem Workspace 适配器。

    可传入一个已构造好的 ``MemoryClient``（``client=``），或用 ``settings=`` /
    ``storage_dir=`` / ``mode=`` / ``embed=`` / ``llm=`` 现场构建。
    """
    backend = DualMemMemory(
        user_id=user_id or "",
        agent_id=agent_id,
        app_id=app_id,
        settings=settings,
        client=client,
        storage_dir=storage_dir,
        mode=mode,
        embed=embed,
        llm=llm,
    )
    return backend.as_workspace(
        workspace_dir=workspace_dir,
        config=config,
        agent_id=agent_id or None,
        app_id=app_id,
        max_recall=max_recall,
        min_score=min_score,
        profile_limit=profile_limit,
        profile_min_score=profile_min_score,
        intention_limit=intention_limit,
        use_experiences=use_experiences,
        frozen_memory=frozen_memory,
    )


# 别名：与 get_backend("agentica") 互补，语义更明确。
build_agentica_workspace = get_agentica_memory_backend
