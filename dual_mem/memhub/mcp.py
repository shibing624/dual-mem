# -*- coding: utf-8 -*-
"""Local stdio MCP proxy for one bearer-authenticated MemHub Core."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from dual_mem.memhub.client import HubClient


def build_hub_mcp(
    *,
    hub_dir: str | Path | None = None,
    client: HubClient | None = None,
):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "MemHub MCP requires: pip install dual-mem[mcp]"
        ) from exc

    core = client or HubClient.from_config(hub_dir)
    mcp = FastMCP("dual-mem-hub")

    @mcp.tool(
        description=(
            "写入当前用户的全局个人记忆。content 与 messages 二选一；"
            "普通会话已由 Hook 自动采集，不要重复提交。"
        )
    )
    async def memory_add(
        content: str = "",
        messages: list[dict] | None = None,
        session_id: str = "",
    ) -> dict:
        job = await core.write(
            content=content,
            messages=messages,
            session_id=session_id,
            request_id=f"hub-mcp:{uuid.uuid4()}",
        )
        return job["result"] or job

    @mcp.tool(description="检索当前用户的画像、事实、意图和派生记忆。")
    async def memory_search(
        query: str,
        limit: int = 10,
    ) -> dict:
        result = await core.recall(query=query, limit=limit, raw_limit=0)
        return result["memories"]

    @mcp.tool(description="检索历史对话原文，用于核对用户之前说过或问过什么。")
    async def conversation_search(
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        result = await core.recall(query=query, limit=0, raw_limit=limit)
        return result["conversations"]

    @mcp.tool(description="查看 Hub Core、记忆和写队列状态。")
    async def memory_status() -> dict:
        return await core.status()

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(prog="dual-mem-hub-mcp")
    parser.add_argument("--hub-dir")
    args = parser.parse_args()
    build_hub_mcp(hub_dir=args.hub_dir).run(transport="stdio")


if __name__ == "__main__":
    main()
