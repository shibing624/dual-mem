# -*- coding: utf-8 -*-
"""`dual-mem hub`: attach hosts to one loopback-first single-writer Core."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import typer

from dual_mem.memhub import hosts as hub_hosts
from dual_mem.memhub import service as hub_service
from dual_mem.memhub.client import HubClient
from dual_mem.memhub.config import (
    CursorStore,
    HubConfig,
    clear_hub_config,
    hub_root,
    hub_settings,
    load_hub_config,
    runtime_status,
    save_hub_config,
)
from dual_mem.memhub.sources import SOURCES, locate_transcript, scan_file, scan_sources
from dual_mem.retrieval.formatter import format_memories

hub_app = typer.Typer(help="MemHub：本地单 Core、跨 CLI 共享个人记忆")
HubDir = typer.Option(None, "--hub-dir", help="Hub 根目录（默认 ~/.dual-mem）")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def make_client(hub_dir: str | None) -> HubClient:
    """Indirection for tests."""
    return HubClient.from_config(hub_dir)


def _echo(data: Any) -> None:
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


def _targets(host: str) -> list[str]:
    if host == "all":
        return list(hub_hosts.HOSTS)
    if host not in hub_hosts.HOSTS:
        raise typer.BadParameter(
            f"unknown host {host!r}; choose from "
            f"{', '.join(hub_hosts.HOSTS)} or all"
        )
    return [host]


@contextmanager
def _hook_errors_logged(hub_dir: str | None, name: str) -> Iterator[None]:
    """Host hooks fail open; diagnostics stay under the local Hub home."""
    try:
        yield
    except Exception:
        log_dir = hub_root(hub_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        logger = logging.getLogger("dual_mem.memhub.hooks")
        handler = logging.FileHandler(log_dir / "hooks.log", encoding="utf-8")
        logger.addHandler(handler)
        logger.exception("%s hook failed", name)
        logger.removeHandler(handler)
        handler.close()


def _spawn_detached(argv: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )


def _validate_base_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise typer.BadParameter("base_url must be an absolute http(s) URL")
    loopback = parsed.hostname in LOOPBACK_HOSTS
    if parsed.scheme != "https" and not loopback:
        raise typer.BadParameter(
            "non-loopback Hub connections require HTTPS; use an SSH/Tailscale "
            "tunnel and a 127.0.0.1 URL"
        )
    return value


@hub_app.command("attach")
def hub_attach(
    base_url: str = typer.Option(..., "--base-url", help="Hub Core URL"),
    api_key: str = typer.Option(..., "--api-key", help="Hub Core Bearer API key"),
    hub_dir: str | None = HubDir,
) -> None:
    """显式连接隧道/远程 Core；本地 Core 无需执行此命令。"""
    normalized = _validate_base_url(base_url)

    async def attach() -> dict[str, Any]:
        async with HubClient(base_url=normalized, api_key=api_key) as client:
            return await client.identity()

    identity = asyncio.run(attach())
    cfg = HubConfig(base_url=normalized, api_key=api_key)
    save_hub_config(cfg, hub_dir)
    _echo(
        {
            "hub_dir": str(hub_root(hub_dir)),
            "config": cfg.public_dict(),
            "identity": identity,
        }
    )


@hub_app.command("detach")
def hub_detach(hub_dir: str | None = HubDir) -> None:
    """移除远程连接（含 connect 建立的 SSH 隧道服务），恢复本机 Core 自动发现。"""
    tunnel = (
        hub_service.uninstall_tunnel()
        if load_hub_config(hub_dir).ssh_target
        else None
    )
    _echo(
        {
            "hub_dir": str(hub_root(hub_dir)),
            "detached": clear_hub_config(hub_dir),
            "tunnel": tunnel,
        }
    )


def _remote_path(path: str) -> str:
    """Quote a remote path for the remote shell, keeping a leading ``~/`` expandable."""
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def read_remote_core(
    target: str, *, remote_hub_dir: str, ssh_args: list[str]
) -> tuple[str, int]:
    """Bearer token and loopback port of the Core running on ``target``.

    One ``ssh`` call reads the server's ``runtime-token`` and ``runtime.json``
    (published by a running Core), so nothing is copied by hand.
    """
    root = _remote_path(remote_hub_dir.rstrip("/") or "~")
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            *ssh_args,
            target,
            f"cat {root}/runtime-token && cat {root}/runtime.json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise typer.BadParameter(
            f"cannot read the Core on {target}: {result.stderr.strip()}. "
            "Is `dual-mem hub install-service` running there, and does "
            "`ssh` reach it without a password prompt?"
        )
    token, _, runtime = result.stdout.partition("\n")
    port = urlparse(json.loads(runtime)["base_url"]).port
    if not token.strip() or port is None:
        raise typer.BadParameter(f"incomplete Core runtime on {target}")
    return token.strip(), port


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _wait_identity(
    base_url: str, api_key: str, *, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    async with HubClient(base_url=base_url, api_key=api_key) as client:
        while True:
            try:
                return await client.identity()
            except httpx.TransportError:
                # The tunnel service is still starting; give up only at the deadline.
                # An HTTP error (wrong token, not a Hub) is final and raises at once.
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.5)


@hub_app.command("connect")
def hub_connect(
    target: str = typer.Argument(..., help="SSH 目标，如 user@server 或 ~/.ssh/config 里的 Host"),
    local_port: int = typer.Option(0, "--local-port", help="本机隧道端口，0 = 自动选空闲端口"),
    remote_hub_dir: str = typer.Option("~/.dual-mem", "--remote-hub-dir", help="服务器上的 Hub 根目录"),
    ssh_arg: list[str] = typer.Option(
        [], "--ssh-arg", help="透传给 ssh 的参数，可重复，如 --ssh-arg=-p --ssh-arg=2222"
    ),
    timeout: float = typer.Option(20.0, "--timeout", help="等待隧道连通的秒数"),
    hub_dir: str | None = HubDir,
) -> None:
    """一条命令连上服务器的 Core：读 token、建常驻 SSH 隧道、写本机 hub.json。

    Core 仍只监听服务器的 127.0.0.1；隧道作为登录服务运行，断线自动重连。
    """
    token, remote_port = read_remote_core(
        target, remote_hub_dir=remote_hub_dir, ssh_args=ssh_arg
    )
    port = local_port or free_local_port()
    tunnel = hub_service.install_tunnel(
        target,
        local_port=port,
        remote_port=remote_port,
        ssh_args=ssh_arg,
        hub_dir=hub_dir,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        identity = asyncio.run(_wait_identity(base_url, token, timeout=timeout))
    except httpx.HTTPError as exc:
        hub_service.uninstall_tunnel()
        raise typer.BadParameter(
            f"Hub Core on {target} not reachable through the tunnel ({exc}); "
            f"see {tunnel['log']}"
        ) from exc
    cfg = HubConfig(base_url=base_url, api_key=token, ssh_target=target)
    save_hub_config(cfg, hub_dir)
    _echo(
        {
            "hub_dir": str(hub_root(hub_dir)),
            "config": cfg.public_dict(),
            "tunnel": tunnel["service_file"],
            "identity": identity,
        }
    )


@hub_app.command("serve")
def hub_serve(
    host: str = typer.Option("127.0.0.1", "--host", help="只允许 loopback 地址"),
    port: int = typer.Option(8765, "--port"),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="DUAL_MEM_HUB_SERVER_API_KEY",
        help="可选固定 Bearer key；默认自动生成 runtime-token",
    ),
    mode: str | None = typer.Option(None, "--mode", help="system1 | dual"),
    digest_idle: float = typer.Option(
        300.0,
        "--digest-idle",
        help="dual 模式：写入队列空闲多少秒后自动 digest()，0 = 关闭",
    ),
    hub_dir: str | None = HubDir,
) -> None:
    """启动唯一 Hub Core；记忆与 Evidence 均留在本机 Hub 根目录。"""
    import uvicorn

    from dual_mem.memhub.server import create_hub_app

    if host not in LOOPBACK_HOSTS:
        # The bearer token travels in plain HTTP; remote clients use a tunnel.
        raise typer.BadParameter(
            "Hub Core listens on loopback only; reach it from other machines "
            "through an SSH/Tailscale tunnel"
        )
    url_host = f"[{host}]" if ":" in host else host
    app = create_hub_app(
        api_key=api_key,
        hub_dir=hub_dir,
        settings=hub_settings(hub_dir, mode=mode),
        published_base_url=f"http://{url_host}:{port}",
        digest_idle_seconds=digest_idle,
    )
    uvicorn.run(app, host=host, port=port)


@hub_app.command("install-service")
def hub_install_service(
    port: int = typer.Option(8765, "--port"),
    hub_dir: str | None = HubDir,
) -> None:
    """开机/登录自动启动 Hub Core（macOS LaunchAgent / Linux systemd --user）。

    服务不继承 shell 环境变量，模型 API key 请写在 ~/.dual_mem/config.yaml。"""
    _echo(hub_service.install_service(hub_dir=hub_dir, port=port))


@hub_app.command("uninstall-service")
def hub_uninstall_service() -> None:
    """停止并移除 Hub Core 自启动服务（不删除记忆数据）。"""
    _echo(hub_service.uninstall_service())


@hub_app.command("extract")
def hub_extract(hub_dir: str | None = HubDir) -> None:
    """对 --raw-only 导入的 Evidence 补跑抽取（后台排队，复用已存的 L0）。"""

    async def run() -> dict[str, Any]:
        async with make_client(hub_dir) as client:
            return await client.extract_raw()

    _echo(asyncio.run(run()))


@hub_app.command("retry")
def hub_retry(hub_dir: str | None = HubDir) -> None:
    """把重试耗尽而失败的写入重新排队。"""

    async def run() -> dict[str, Any]:
        async with make_client(hub_dir) as client:
            return await client.retry_failed()

    _echo(asyncio.run(run()))


@hub_app.command("status")
def hub_status(hub_dir: str | None = HubDir) -> None:
    """Core 记忆/队列状态，以及本机扫描进度和宿主接入状态。"""

    async def remote() -> dict[str, Any]:
        async with make_client(hub_dir) as client:
            return await client.status()

    data = asyncio.run(remote())
    with CursorStore(hub_dir) as cursors:
        scanned = Counter()
        for cursor in cursors.all():
            scanned[cursor.source] += cursor.turns
    attachment = load_hub_config(hub_dir)
    data.update(
        {
            "connection": (
                attachment.public_dict()
                if attachment.base_url
                else {"mode": "local", "runtime": runtime_status(hub_dir)}
            ),
            "turns_scanned": dict(scanned),
            "hosts": hub_hosts.hosts_status(),
        }
    )
    _echo(data)


@hub_app.command("install-host")
def hub_install_host(
    host: str = typer.Argument("all", help=f"{' | '.join(hub_hosts.HOSTS)} | all"),
    hub_dir: str | None = HubDir,
) -> None:
    """安装 Hub MCP 代理、recall skill 与采集 Hook。"""
    _echo(
        [
            hub_hosts.install_host(item, hub_dir=hub_dir)
            for item in _targets(host)
        ]
    )


@hub_app.command("uninstall-host")
def hub_uninstall_host(
    host: str = typer.Argument("all", help=f"{' | '.join(hub_hosts.HOSTS)} | all"),
) -> None:
    """移除 Hub 写入的配置，其它宿主配置保持不动。"""
    _echo([hub_hosts.uninstall_host(item) for item in _targets(host)])


@hub_app.command("scan")
def hub_scan(
    source: str = typer.Option(
        "all",
        "--source",
        "-s",
        help=f"{' | '.join(SOURCES)} | all",
    ),
    file: Path | None = typer.Option(None, "--file", help="只扫一个会话文件"),
    complete: bool = typer.Option(False, "--complete", help="最后一轮已经结束"),
    raw_only: bool = typer.Option(False, "--raw-only", help="只保留原始 Evidence，之后可用 hub extract 补抽取"),
    limit: int = typer.Option(0, "--limit", help="最多写入轮数，0 = 不限"),
    batch: int = typer.Option(5, "--batch", help="同会话每次提交的轮数"),
    idle: int = typer.Option(300, "--idle", help="静默多少秒视为一轮结束"),
    hub_dir: str | None = HubDir,
) -> None:
    """按本机游标增量读取会话，并提交给唯一 Hub Core。"""
    selected = list(SOURCES) if source == "all" else [source]
    if not set(selected) <= set(SOURCES):
        raise typer.BadParameter(f"unknown source {source!r}")

    async def run() -> Any:
        async with make_client(hub_dir) as client:
            options = {
                "raw_only": raw_only,
                "batch_turns": batch,
                "idle_seconds": idle,
            }
            if file is None:
                return await scan_sources(
                    client,
                    hub_dir=hub_dir,
                    sources=selected,
                    max_turns=limit,
                    **options,
                )
            with CursorStore(hub_dir) as cursors:
                return {
                    selected[0]: await scan_file(
                        client,
                        source=selected[0],
                        path=file,
                        cursors=cursors,
                        complete=complete,
                        max_turns=limit,
                        **options,
                    )
                }

    _echo(asyncio.run(run()))


@hub_app.command("capture")
def hub_capture(
    source: str = typer.Argument(..., help=f"{' | '.join(SOURCES)}"),
    hub_dir: str | None = HubDir,
) -> None:
    """Hook 入口：定位 transcript 后后台扫描，立即返回宿主。"""
    with _hook_errors_logged(hub_dir, f"capture {source}"):
        payload = json.loads(sys.stdin.read() or "{}")
        transcript = locate_transcript(source, payload)
        if transcript is None:
            return
        argv = [
            sys.executable,
            "-m",
            "dual_mem.cli",
            "hub",
            "scan",
            "--source",
            source,
            "--file",
            str(transcript),
            "--complete",
        ]
        if hub_dir:
            argv += ["--hub-dir", hub_dir]
        _spawn_detached(argv, hub_root(hub_dir) / "logs" / "capture.log")


@hub_app.command("hook-search")
def hub_hook_search(
    limit: int = typer.Option(8, "--limit"),
    hub_dir: str | None = HubDir,
) -> None:
    """UserPromptSubmit：从 Core 召回并注入 additionalContext。"""
    with _hook_errors_logged(hub_dir, "hook-search"):
        prompt = json.loads(sys.stdin.read() or "{}").get("prompt", "").strip()
        if not prompt:
            return

        async def search() -> dict[str, Any]:
            async with make_client(hub_dir) as client:
                return await client.recall(query=prompt, limit=limit, raw_limit=3)

        result = asyncio.run(search())
        context = format_memories(result["memories"]).strip()
        if result["conversations"]:
            raw = "\n".join(
                f"- [{item['session_id']}] {item['content'].strip()}"
                for item in result["conversations"]
            )
            context = f"{context}\n\n历史对话：\n{raw}".strip()
        if context:
            # Tagged so strip_memory_injection drops it if a host echoes it back.
            context = f"<relevant-memories>\n{context}\n</relevant-memories>"
            typer.echo(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "UserPromptSubmit",
                            "additionalContext": context,
                        }
                    },
                    ensure_ascii=False,
                )
            )


@hub_app.command("recall")
def hub_recall(
    query: str = typer.Argument(...),
    limit: int = typer.Option(10, "--limit"),
    raw_limit: int = typer.Option(5, "--raw-limit"),
    json_out: bool = typer.Option(False, "--json"),
    hub_dir: str | None = HubDir,
) -> None:
    """同时检索当前记忆和历史对话 Evidence。"""

    async def search() -> dict[str, Any]:
        async with make_client(hub_dir) as client:
            return await client.recall(
                query=query,
                limit=limit,
                raw_limit=raw_limit,
            )

    result = asyncio.run(search())
    if json_out:
        _echo(result)
        return
    sections = []
    memories = format_memories(result["memories"]).strip()
    if memories:
        sections.append(f"## 记忆\n{memories}")
    if result["conversations"]:
        sections.append(
            "## 历史对话\n"
            + "\n".join(
                f"- [{item['session_id']}] {item['content'].strip()}"
                for item in result["conversations"]
            )
        )
    typer.echo("\n\n".join(sections) or "没有相关记忆。")
