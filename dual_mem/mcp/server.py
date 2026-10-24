# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP server — registers the same memory_* tools as REST (MemoryOperations).
"""
import argparse
import sys

from dual_mem.operations import MemoryOperations
from dual_mem.client import MemoryClient
from dual_mem.config import Settings

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_SEARCH_DOC = (
    "语义检索，结果按 profile / proactive / normal 三路分组；"
    "有演化历史的记忆带 evolution_chain（最新→最旧）。"
    "procedure_limit>0 时另返回 procedures 组：coding agent 工具轨迹里沉淀的工程经验"
    "（task / solution / files）。"
)


def build_mcp(
    *,
    client: MemoryClient | None = None,
    ops: MemoryOperations | None = None,
):
    """Register dual-mem memory tools on an MCP server (mcp 2.x ``MCPServer``).

    Transport, host and port are chosen at run time (``run_server``), not here.
    """
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        raise ImportError(
            "MCP support requires the optional dependency. Install with: pip install dual-mem[mcp]"
        ) from exc

    if ops is None:
        if client is None:
            client = MemoryClient(settings=Settings())
        ops = MemoryOperations(client)

    mcp = MCPServer("dual-mem")

    @mcp.tool(
        description=(
            "写入记忆。content 与 messages 二选一；"
            "messages 为 [{role, content}, ...] 多轮对话，可含 role=tool 与 assistant "
            "tool_calls（工具轨迹会另外沉淀 procedure 经验）。必填 user_id；"
            "app_id 省略则用 default_app_id；workspace_id / branch 是 procedure 的版本边界。"
        )
    )
    async def memory_add(
        user_id: str,
        app_id: str | None = None,
        content: str = "",
        messages: list[dict] | None = None,
        agent_id: str = "",
        session_id: str = "",
        workspace_id: str | None = None,
        branch: str | None = None,
    ) -> dict:
        return await ops.memory_add(
            user_id=user_id,
            app_id=app_id,
            content=content,
            messages=messages,
            agent_id=agent_id,
            session_id=session_id,
            workspace_id=workspace_id,
            branch=branch,
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
        include_derived: bool | None = None,
        as_of: int | None = None,
        raw_limit: int = 0,
        procedure_limit: int = 0,
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
            as_of=as_of,
            raw_limit=raw_limit,
            procedure_limit=procedure_limit,
        )

    @mcp.tool(description="检索 L0 原文及保留的历史，用于核对原话/时间/来源。")
    async def conversation_search(
        query: str,
        user_id: str,
        app_ids: list[str] | None = None,
        agent_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> dict:
        return await ops.conversation_search(
            query=query,
            user_id=user_id,
            app_ids=app_ids,
            agent_ids=agent_ids,
            limit=limit,
            min_score=min_score,
        )

    @mcp.tool(description="列出某 scope 的当前记忆。include_history=true 时连同已归档版本。")
    async def memory_list(
        user_id: str,
        app_id: str | None = None,
        agent_id: str = "",
        limit: int = 100,
        include_history: bool = False,
    ) -> list[dict]:
        return await ops.memory_list(
            user_id=user_id,
            app_id=app_id,
            agent_id=agent_id,
            limit=limit,
            include_history=include_history,
        )

    @mcp.tool(description="按 memory_id 获取单条；不存在返回 null。")
    async def memory_get(memory_id: str) -> dict | None:
        return await ops.memory_get(memory_id)

    @mcp.tool(description="追加一版更正并返回新 memory_id；旧 id 仍可 get，is_latest=false。")
    async def memory_update(memory_id: str, content: str) -> dict:
        return await ops.memory_update(memory_id, content)

    @mcp.tool(description="归档一条当前记忆，不物理删除。删非当前版本返回 409。")
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

    @mcp.tool(description="dual 模式生成 L3_DERIVED。system1 返回 processed=0，不会自动后台巩固。")
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
        build_mcp().run(transport="stdio")
    elif transport == "streamable-http":
        print(
            f"dual-mem MCP (streamable-http): starting http://{host}:{port}/mcp",
            file=sys.stderr,
        )
        build_mcp().run(transport="streamable-http", host=host, port=port)
    else:
        raise ValueError(f"unknown MCP transport {transport!r}: stdio | streamable-http")


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
