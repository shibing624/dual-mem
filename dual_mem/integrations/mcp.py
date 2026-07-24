# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP 后端入口 — 复用 dual_mem.mcp.server 的 FastMCP builder。
"""
from __future__ import annotations

from typing import Any, Optional

from dual_mem.client import MemoryClient
from dual_mem.config import Settings


def build_mcp_backend(
    *,
    client: Optional[MemoryClient] = None,
    settings: Optional[Settings] = None,
    storage_dir: Optional[str] = None,
    mode: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    embed: Any = None,
    llm: Any = None,
) -> Any:
    """Build the dual-mem FastMCP server instance.

    Constructs a ``MemoryClient`` from the provided settings / storage_dir / mode /
    injected ``embed`` / ``llm`` when ``client`` is not given, then hands it to the
    shared ``build_mcp`` so the integration layer and the standalone ``dual-mem-mcp``
    CLI use one implementation.
    """
    from dual_mem.mcp.server import build_mcp

    if client is None:
        if settings is None:
            overrides: dict[str, Any] = {}
            if storage_dir is not None:
                overrides["storage_dir"] = storage_dir
            if mode is not None:
                overrides["mode"] = mode
            settings = Settings(**overrides)
        client = MemoryClient(settings=settings, embed=embed, llm=llm)
    return build_mcp(client=client, host=host, port=port)
