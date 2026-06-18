# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP server exposing dual-mem memory tools (add/search/get/list/delete) over
FastMCP, backed by a MemoryClient. Supports stdio and Streamable HTTP transports and ships a
console entry point (dual-mem-mcp) for uvx.
"""
import argparse

from mcp.server.fastmcp import FastMCP

from dual_mem.client import MemoryClient
from dual_mem.config import Settings

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_GROUP_DOC = (
    "搜索结果按三路分组返回：profile 是稳定的用户画像/身份/模式记忆；"
    "proactive 是推断出的用户意图（仅 ultra 模式非空）；normal 是普通事实与知识记忆。"
    "若某条记忆经历过演化更新，会带 evolution_chain 字段（按 最新→最旧 排序），"
    "代表同一记忆的多个历史版本。"
)


def build_mcp(
    *, client: MemoryClient | None = None, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> FastMCP:
    """Build a FastMCP server registering the dual-mem memory tools.

    host/port only take effect for the Streamable HTTP transport (path defaults to /mcp).
    """
    if client is None:
        client = MemoryClient(settings=Settings())

    mcp = FastMCP("dual-mem", host=host, port=port)

    @mcp.tool(
        description=(
            "写入一条记忆。content 为纯文本内容（与 messages 二选一）；"
            "app_id+user_id 是必填的归属标识。返回 memory_id。"
        )
    )
    async def memory_add(
        content: str,
        app_id: str,
        user_id: str,
        agent_id: str = "",
        session_id: str = "",
    ) -> dict:
        return await client.add(
            content=content,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

    @mcp.tool(
        description=(
            "语义检索记忆。" + _GROUP_DOC + " 用 query 描述你想回忆的信息，"
            "app_ids+user_id 限定检索范围。"
        )
    )
    async def memory_search(
        query: str,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.4,
        intention_limit: int = 0,
    ) -> dict:
        return await client.search(
            query=query,
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            limit=limit,
            min_score=min_score,
            intention_limit=intention_limit,
        )

    @mcp.tool(description="按 memory_id 获取单条记忆，不存在返回 null。")
    async def memory_get(memory_id: str) -> dict | None:
        return await client.get(memory_id)

    @mcp.tool(description="列出某 app_id+user_id（可选 agent_id）下的记忆。")
    async def memory_list(
        app_id: str,
        user_id: str,
        agent_id: str = "",
        limit: int = 100,
    ) -> list[dict]:
        return await client.list(
            app_id=app_id, user_id=user_id, agent_id=agent_id, limit=limit
        )

    @mcp.tool(description="按 memory_id 删除单条记忆（幂等）。")
    async def memory_delete(memory_id: str) -> dict:
        return await client.delete(memory_id)

    return mcp


def run_server(
    *, transport: str = "stdio", host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> None:
    """Run the MCP server with the given transport (stdio or streamable-http)."""
    mcp = build_mcp(host=host, port=port)
    mcp.run(transport=transport)


def main() -> None:
    """Console entry point (dual-mem-mcp): parse transport flags and run the server."""
    parser = argparse.ArgumentParser(
        prog="dual-mem-mcp", description="dual-mem MCP server (stdio / Streamable HTTP)"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="传输方式：stdio（默认，供 Cursor/Claude Desktop 本地拉起）或 streamable-http（暴露 /mcp）",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="streamable-http 监听地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="streamable-http 监听端口")
    args = parser.parse_args()
    run_server(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
