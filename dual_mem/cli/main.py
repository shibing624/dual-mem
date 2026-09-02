# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Typer CLI — memory CRUD/search/scope commands over MemoryClient.
"""
import asyncio
import json
from pathlib import Path

import typer

from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.retrieval.formatter import format_memories

app = typer.Typer(help="dual-mem 分层记忆 SDK 命令行")


def make_client(mode: str | None = None) -> MemoryClient:
    return MemoryClient(mode=mode)


def _echo_json(data) -> None:
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


def _run(coro):
    return asyncio.run(coro)


def _parse_messages_json(raw: str, *, source: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        typer.echo(f"{source} 不是合法 JSON: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not isinstance(data, list) or not data:
        typer.echo(f"{source} 须为非空 JSON 数组，如 [{{\"role\":\"user\",\"content\":\"...\"}}]", err=True)
        raise typer.Exit(code=1)
    for i, item in enumerate(data):
        if not isinstance(item, dict) or not str(item.get("content", "")).strip():
            typer.echo(f"{source} 第 {i} 项须为含 content 的对象", err=True)
            raise typer.Exit(code=1)
    return data


def _load_messages(*, messages_file: str, messages_json: str) -> list[dict]:
    if messages_file and messages_json:
        typer.echo("--messages-file 与 --messages-json 二选一", err=True)
        raise typer.Exit(code=1)
    if messages_file:
        path = Path(messages_file)
        if not path.is_file():
            typer.echo(f"文件不存在: {messages_file}", err=True)
            raise typer.Exit(code=1)
        return _parse_messages_json(path.read_text(encoding="utf-8"), source=messages_file)
    return _parse_messages_json(messages_json, source="--messages-json")


@app.command()
def add(
    content: str = typer.Option("", "--content"),
    messages_file: str = typer.Option(
        "",
        "--messages-file",
        help="多轮对话 JSON 文件，格式 [{\"role\":\"user\",\"content\":\"...\"}, ...]",
    ),
    messages_json: str = typer.Option(
        "",
        "--messages-json",
        help="多轮对话 JSON 字符串（与 --messages-file 二选一）",
    ),
    app_id: str | None = typer.Option(
        None,
        "--app-id",
        help="省略则使用 config 中 default_app_id（默认 default）",
    ),
    user_id: str = typer.Option(..., "--user-id"),
    agent_id: str = typer.Option("", "--agent-id"),
    session_id: str = typer.Option("", "--session-id"),
    mode: str | None = typer.Option(None, "--mode", help="system1 | dual（默认取配置）"),
):
    """写入记忆（MemoryClient.add）。content 与 messages 二选一。"""
    has_content = bool(content.strip())
    has_messages = bool(messages_file or messages_json)
    if has_content == has_messages:
        typer.echo("请提供 --content 或 --messages-file / --messages-json（二选一）", err=True)
        raise typer.Exit(code=1)
    client = make_client(mode)
    if has_messages:
        messages = _load_messages(messages_file=messages_file, messages_json=messages_json)
        result = _run(
            client.add(
                messages=messages,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )
        )
    else:
        result = _run(
            client.add(
                content=content,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )
        )
    _echo_json(result.to_dict())


@app.command()
def search(
    query: str = typer.Argument(...),
    app_id: str | None = typer.Option(
        None,
        "--app-id",
        help="省略则使用 config 中 default_app_id",
    ),
    user_id: str = typer.Option(..., "--user-id"),
    limit: int = typer.Option(10, "--limit"),
    json_out: bool = typer.Option(False, "--json", help="输出原始 JSON 而非格式化文本"),
    mode: str | None = typer.Option(None, "--mode", help="system1 | dual"),
):
    """语义检索（MemoryClient.search）。"""
    client = make_client(mode)
    result = _run(
        client.search(
            query=query,
            app_ids=[app_id] if app_id is not None else None,
            user_id=user_id,
            limit=limit,
        )
    )
    if json_out:
        _echo_json(result.to_dict())
    else:
        typer.echo(format_memories(result.memories.to_dict()))


@app.command(name="search-conversation")
def search_conversation(
    query: str = typer.Argument(...),
    app_id: str | None = typer.Option(
        None,
        "--app-id",
        help="省略则使用 config 中 default_app_id",
    ),
    user_id: str = typer.Option(..., "--user-id"),
    limit: int = typer.Option(10, "--limit"),
    json_out: bool = typer.Option(False, "--json", help="输出原始 JSON 而非格式化文本"),
    mode: str | None = typer.Option(None, "--mode", help="system1 | dual"),
):
    """检索 L1 原文（MemoryClient.search_conversation）。"""
    client = make_client(mode)
    result = _run(
        client.search_conversation(
            query=query,
            app_ids=[app_id] if app_id is not None else None,
            user_id=user_id,
            limit=limit,
        )
    )
    if json_out:
        _echo_json(result.to_dict())
    else:
        typer.echo(format_memories(result.memories.to_dict()))


@app.command(name="list")
def list_memories(
    app_id: str | None = typer.Option(
        None,
        "--app-id",
        help="省略则使用 config 中 default_app_id",
    ),
    user_id: str = typer.Option(..., "--user-id"),
    agent_id: str = typer.Option("", "--agent-id"),
    limit: int = typer.Option(100, "--limit"),
):
    """列出 scope 下 ACTIVE 记忆（MemoryClient.list）。"""
    client = make_client()
    items = _run(
        client.list(app_id=app_id, user_id=user_id, agent_id=agent_id, limit=limit)
    )
    _echo_json([item.to_dict() for item in items])


@app.command()
def get(memory_id: str = typer.Argument(...)):
    """获取单条记忆（MemoryClient.get）。"""
    client = make_client()
    item = _run(client.get(memory_id))
    _echo_json(item.to_dict() if item is not None else None)


@app.command()
def update(
    memory_id: str = typer.Argument(...),
    content: str = typer.Option(..., "--content"),
):
    """更新记忆正文（MemoryClient.update）。"""
    client = make_client()
    result = _run(client.update(memory_id, content))
    _echo_json(result.to_dict())


@app.command()
def delete(memory_id: str = typer.Argument(...)):
    """删除单条记忆（MemoryClient.delete）。"""
    client = make_client()
    result = _run(client.delete(memory_id))
    _echo_json(result.to_dict())


@app.command(name="delete-scope")
def delete_scope(
    app_id: str | None = typer.Option(
        None,
        "--app-id",
        help="省略则使用 config 中 default_app_id",
    ),
    user_id: str | None = typer.Option(None, "--user-id"),
    agent_id: str | None = typer.Option(None, "--agent-id"),
    confirm: bool = typer.Option(False, "--confirm", help="必须显式确认批量删除"),
):
    """按 scope 批量删除（MemoryClient.delete_bulk）。"""
    client = make_client()
    result = _run(
        client.delete_bulk(
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            confirm=confirm,
        )
    )
    _echo_json(result.to_dict())


@app.command(name="list-scopes")
def list_scopes(
    app_id: str | None = typer.Option(None, "--app-id"),
    limit: int = typer.Option(5000, "--limit"),
):
    """列出已有记忆 scope（MemoryClient.list_scopes）。"""
    client = make_client()
    scopes = _run(client.list_scopes(app_id=app_id, limit=limit))
    _echo_json([s.to_dict() for s in scopes])


@app.command()
def digest():
    """触发 System2 深度记忆巩固（MemoryClient.digest，dual 模式）。"""
    client = make_client()
    result = _run(client.digest())
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
    """启动 MCP 服务。"""
    from dual_mem.mcp.server import run_server

    run_server(transport=transport, host=host, port=port)


if __name__ == "__main__":
    app()
