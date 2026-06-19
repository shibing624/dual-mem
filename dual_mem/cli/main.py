# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Typer CLI for the dual-mem SDK with commands to add/search/list/get/delete
memories, trigger System2 digest, and serve the REST API or MCP server. SDK results are
dataclasses; this layer flattens them with .to_dict() for JSON output and friendly display.
"""
import asyncio
import json

import typer

from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.retrieval.formatter import format_memories

app = typer.Typer(help="dual-mem 分层记忆 SDK 命令行")


def make_client(mode: str | None = None) -> MemoryClient:
    """Construct a MemoryClient for the given mode (or the configured default)."""
    return MemoryClient(mode=mode)


def _echo_json(data) -> None:
    """Pretty-print data as UTF-8 JSON to stdout."""
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


@app.command()
def add(
    content: str = typer.Option(..., "--content"),
    app_id: str = typer.Option(..., "--app-id"),
    user_id: str = typer.Option(..., "--user-id"),
    agent_id: str = typer.Option("", "--agent-id"),
    mode: str | None = typer.Option(None, "--mode", help="emb | system1 | dual（默认取配置）"),
):
    """写入一条记忆。"""
    client = make_client(mode)
    result = asyncio.run(
        client.add(content=content, app_id=app_id, user_id=user_id, agent_id=agent_id)
    )
    _echo_json(result.to_dict())


@app.command()
def search(
    query: str = typer.Argument(...),
    app_id: str = typer.Option(..., "--app-id"),
    user_id: str = typer.Option(..., "--user-id"),
    limit: int = typer.Option(10, "--limit"),
    mode: str | None = typer.Option(None, "--mode", help="emb | system1 | dual（默认取配置）"),
):
    """语义检索记忆并友好展示。"""
    client = make_client(mode)
    result = asyncio.run(
        client.search(query=query, app_ids=[app_id], user_id=user_id, limit=limit)
    )
    typer.echo(format_memories(result.memories.to_dict()))


@app.command(name="list")
def list_memories(
    app_id: str = typer.Option(..., "--app-id"),
    user_id: str = typer.Option(..., "--user-id"),
    agent_id: str = typer.Option("", "--agent-id"),
    limit: int = typer.Option(100, "--limit"),
):
    """列出记忆。"""
    client = make_client()
    items = asyncio.run(
        client.list(app_id=app_id, user_id=user_id, agent_id=agent_id, limit=limit)
    )
    _echo_json([item.to_dict() for item in items])


@app.command()
def get(memory_id: str = typer.Argument(...)):
    """获取单条记忆。"""
    client = make_client()
    item = asyncio.run(client.get(memory_id))
    _echo_json(item.to_dict() if item is not None else None)


@app.command()
def delete(memory_id: str = typer.Argument(...)):
    """删除单条记忆。"""
    client = make_client()
    result = asyncio.run(client.delete(memory_id))
    _echo_json(result.to_dict())


@app.command()
def digest():
    """触发 System2 后台沉淀（仅 dual 模式有效）。"""
    client = make_client()
    result = asyncio.run(client.digest())
    _echo_json(result.to_dict())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """启动 REST API 服务。"""
    import uvicorn

    from dual_mem.api import create_app

    uvicorn.run(create_app(settings=Settings()), host=host, port=port)


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", help="stdio | streamable-http"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
):
    """启动 MCP 服务（stdio 或 streamable-http，后者暴露 /mcp 端点）。"""
    from dual_mem.mcp.server import run_server

    run_server(transport=transport, host=host, port=port)


if __name__ == "__main__":
    app()
