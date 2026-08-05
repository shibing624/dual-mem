# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP server — registers the same memory_* tools as REST (MemoryOperations).
"""
import argparse
import sys
from typing import Literal, cast

from dual_mem.api.operations import MemoryOperations
from dual_mem.client import MemoryClient
from dual_mem.config import Settings

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_SEARCH_DOC = (
    "语义检索，结果按 profile / proactive / normal 三路分组；"
    "有演化历史的记忆带 evolution_chain（最新→最旧）。"
)


def build_mcp(
    *,
    client: MemoryClient | None = None,
    ops: MemoryOperations | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
):
    """Register dual-mem memory tools on a FastMCP instance."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "MCP support requires the optional dependency. Install with: pip install dual-mem[mcp]"
        ) from exc

    if ops is None:
        if client is None:
            client = MemoryClient(settings=Settings())
        ops = MemoryOperations(client)

    mcp = FastMCP("dual-mem", host=host, port=port)

    @mcp.tool(
        description=(
            "写入记忆。content 与 messages 二选一；"
            "messages 为 [{role, content}, ...] 多轮对话。必填 user_id；"
            "app_id 省略则用 default_app_id。"
        )
    )
    async def memory_add(
        user_id: str,
        app_id: str | None = None,
        content: str = "",
        messages: list[dict] | None = None,
        agent_id: str = "",
        session_id: str = "",
    ) -> dict:
        return await ops.memory_add(
            user_id=user_id,
            app_id=app_id,
            content=content,
            messages=messages,
            agent_id=agent_id,
            session_id=session_id,
        )

    @mcp.tool(description=f"语义检索。{_SEARCH_DOC}")
    async def memory_search(
        query: str,
        user_id: str,
        app_ids: list[str] | None = None,
        agent_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.4,
        intention_limit: int = 0,
        include_derived: bool = True,
    ) -> dict:
        return await ops.memory_search(
            query=query,
            user_id=user_id,
            app_ids=app_ids,
            agent_ids=agent_ids,
            limit=limit,
            min_score=min_score,
            intention_limit=intention_limit,
            include_derived=include_derived,
        )

    @mcp.tool(description="列出某 scope 下 ACTIVE 记忆。")
    async def memory_list(
        user_id: str,
        app_id: str | None = None,
        agent_id: str = "",
        limit: int = 100,
    ) -> list[dict]:
        return await ops.memory_list(
            user_id=user_id,
            app_id=app_id, agent_id=agent_id, limit=limit
        )

    @mcp.tool(description="按 memory_id 获取单条；不存在返回 null。")
    async def memory_get(memory_id: str) -> dict | None:
        return await ops.memory_get(memory_id)

    @mcp.tool(description="更新记忆正文并重新 embedding。")
    async def memory_update(memory_id: str, content: str) -> dict:
        return await ops.memory_update(memory_id, content)

    @mcp.tool(description="删除单条记忆。")
    async def memory_delete(memory_id: str) -> dict:
        return await ops.memory_delete(memory_id)

    @mcp.tool(
        description="按 scope 批量删除。confirm 必须为 true；app_id 省略则用 default_app_id。"
    )
    async def memory_delete_scope(
        confirm: bool,
        app_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict:
        return await ops.memory_delete_scope(
            confirm=confirm,
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
        )

    @mcp.tool(description="列出存储中已有的记忆 scope（app_id + user_id + agent_id）。")
    async def memory_list_scopes(
        app_id: str | None = None,
        limit: int = 5000,
    ) -> list[dict]:
        return await ops.memory_list_scopes(app_id=app_id, limit=limit)

    @mcp.tool(description="触发 System2 深度记忆巩固（dual 模式）。")
    async def memory_digest() -> dict:
        return await ops.memory_digest()

    return mcp


def run_server(
    *, transport: str = "stdio", host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> None:
    """Run the MCP server with the given transport (stdio or streamable-http)."""

    if transport == "stdio":
        # stdio MCP uses stdout for JSON-RPC — never log there; one stderr line only.
        print(
            "dual-mem MCP (stdio): ready, waiting for client on stdin "
            "(no further console output — normal)",
            file=sys.stderr,
        )
    else:
        print(
            f"dual-mem MCP ({transport}): starting http://{host}:{port}",
            file=sys.stderr,
        )

    mcp = build_mcp(host=host, port=port)
    mcp.run(transport=cast(Literal["stdio", "sse", "streamable-http"], transport))


def main() -> None:
    """Console entry point (dual-mem-mcp)."""
    parser = argparse.ArgumentParser(
        prog="dual-mem-mcp", description="dual-mem MCP server (stdio / Streamable HTTP)"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    run_server(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
