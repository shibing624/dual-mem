# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 社区生态集成层。

把 dual_mem 的 MemoryClient 适配到各类 Agent 框架 / 协议：

- ``mcp``         : FastMCP server（dual-mem-mcp）
- ``hermes``      : Hermes Agent 的 MemoryProvider 插件（被动 prefetch + 异步 sync_turn）
- ``openclaw``    : OpenClaw 的 MemoryProvider 插件（与 Hermes 同源契约）
- ``agentica``    : agentica 框架的 Memory 模块（add_messages / retrieve）+ Workspace 适配器
- ``claude_code`` : Claude Code Hook（UserPromptSubmit / Stop 子命令）

所有适配器都基于 ``MemoryBackend``（对 MemoryClient 的薄封装）+ ``AsyncRunner``
（让异步 client 能被 Hermes/OpenClaw 的同步 hook 回调驱动），并共享同一套
``<relevant-memories>`` 注入块格式化逻辑（借鉴 hy_memory）。
"""
from __future__ import annotations

from typing import Any

from dual_mem.integrations._base import (
    AsyncRunner,
    MemoryBackend,
    format_memories_for_prompt,
)

from dual_mem.integrations.agentica import (
    DualMemMemory,
    DualMemWorkspace,
    build_agentica_workspace,
    get_agentica_memory_backend,
)

__all__ = [
    "MemoryBackend",
    "AsyncRunner",
    "format_memories_for_prompt",
    "DualMemMemory",
    "DualMemWorkspace",
    "get_agentica_memory_backend",
    "build_agentica_workspace",
    "get_backend",
    "list_backends",
]


def list_backends() -> list[str]:
    """Return the names of all supported ecosystem backends."""
    return ["mcp", "hermes", "openclaw", "agentica", "claude_code"]


def get_backend(name: str, **kwargs: Any) -> Any:
    """Build an ecosystem adapter by name.

    Lazy-imports the backend submodule so merely importing
    ``dual_mem.integrations`` never pulls optional framework deps.
    """
    name = name.strip().lower()
    if name == "mcp":
        import dual_mem.integrations.mcp as mod
        return mod.build_mcp_backend(**kwargs)
    if name == "hermes":
        import dual_mem.integrations.hermes as mod
        return mod.DualMemHermesProvider(**kwargs)
    if name == "openclaw":
        import dual_mem.integrations.openclaw as mod
        return mod.DualMemOpenClawProvider(**kwargs)
    if name == "agentica":
        import dual_mem.integrations.agentica as mod
        return mod.DualMemMemory(**kwargs)
    if name == "claude_code":
        import dual_mem.integrations.claude_code as mod
        return mod
    raise ValueError(
        f"Unknown backend {name!r}. Valid: {', '.join(list_backends())}."
    )
