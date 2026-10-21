# -*- coding: utf-8 -*-
"""MemHub Core, transcript intake and host wiring tests."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from dual_mem import MemoryClient
from dual_mem.cli import hub as cli_hub
from dual_mem.cli import main as cli_main
from dual_mem.config import Settings
from dual_mem.memhub import (
    HOSTS,
    CursorStore,
    HubClient,
    HubConfig,
    build_turns,
    default_source_roots,
    hosts_status,
    install_host,
    save_hub_config,
    scan_file,
    scan_sources,
    uninstall_host,
)
from dual_mem.memhub.server import HUB_USER_ID, create_hub_app
from dual_mem.memhub.sources import SOURCE_SPECS
from dual_mem.memhub.config import resolve_hub_config
from dual_mem.types import Layer

runner = CliRunner()


def _memory_client(
    root: Path,
    fake_embed,
    fake_llm,
    fake_jev,
) -> MemoryClient:
    settings = Settings(storage_dir=str(root / "store"), mode="system1")
    return MemoryClient(
        settings=settings,
        embed=fake_embed,
        llm=fake_llm,
        jev=fake_jev,
    )


@asynccontextmanager
async def _attached_core(root: Path, memory_client: MemoryClient):
    app = create_hub_app(api_key="secret", hub_dir=root, client=memory_client)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with HubClient(
            base_url="http://127.0.0.1:8765",
            api_key="secret",
            transport=transport,
        ) as remote:
            yield app, remote


def _jsonl(path: Path, records: list[dict], *, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _hermes_db(path: Path, rows: list[tuple[str, str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT, timestamp REAL, "
        "active INTEGER DEFAULT 1, _compressed_summary INTEGER DEFAULT 0)"
    )
    conn.executemany(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def _seed_transcripts(home: Path) -> dict[str, Path]:
    roots = default_source_roots(home)
    _jsonl(
        roots["agentica"] / "default" / "proj" / "ag-1.jsonl",
        [
            {
                "type": "user",
                "session_id": "ag-1",
                "timestamp": "2026-10-01T10:00:00Z",
                "content": "我在研究强化学习 "
                "<relevant-memories>old echo</relevant-memories>",
            },
            {"type": "assistant", "session_id": "ag-1", "content": "中间步骤"},
            {"type": "tool", "content": "search result"},
            {
                "type": "assistant",
                "session_id": "ag-1",
                "content": "推荐先读 Sutton 的书",
            },
        ],
    )
    _jsonl(
        roots["claude-code"] / "proj" / "cc-1.jsonl",
        [
            {
                "type": "user",
                "sessionId": "cc-1",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "<system-reminder>x</system-reminder>我常用 Python 3.12",
                        }
                    ]
                },
            },
            {
                "type": "assistant",
                "sessionId": "cc-1",
                "message": {"content": [{"type": "text", "text": "好的"}]},
            },
        ],
    )
    _jsonl(
        roots["codex"] / "2026" / "10" / "03" / "rollout-x.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "cx-1"}},
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "item": {
                        "type": "UserMessage",
                        "content": [{"type": "text", "text": "我的网站是 mulanai.com"}],
                    },
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "item": {
                        "type": "AgentMessage",
                        "content": [{"type": "Text", "text": "记住了"}],
                    },
                },
            },
        ],
    )
    _jsonl(
        roots["cursor"] / "proj" / "agent-transcripts" / "cu-1" / "cu-1.jsonl",
        [
            {
                "role": "user",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "<user_query>喜欢简洁的代码</user_query>",
                        }
                    ]
                },
            },
            {
                "role": "assistant",
                "message": {"content": [{"type": "text", "text": "明白"}]},
            },
        ],
    )
    _hermes_db(
        roots["hermes"],
        [
            ("he-1", "user", "周末喜欢爬山", 1791000000.0),
            ("he-1", "assistant", "不错", 1791000001.0),
        ],
    )
    return roots


def test_readers_keep_only_real_dialogue(tmp_path: Path) -> None:
    roots = _seed_transcripts(tmp_path)
    turns = {}
    for name, spec in SOURCE_SPECS.items():
        (path,) = spec.discover(roots[name])
        messages, _ = spec.read(path, 0)
        done, open_turn = build_turns(messages)
        turns[name] = [*done, open_turn]
    assert turns["agentica"][0].assistant == "推荐先读 Sutton 的书"
    assert turns["agentica"][0].user == "我在研究强化学习"
    assert turns["claude-code"][0].user == "我常用 Python 3.12"
    assert turns["codex"][0].session_id == "cx-1"
    assert turns["cursor"][0].user == "喜欢简洁的代码"
    assert turns["hermes"][0].at == 1791000000


def test_build_turns_merges_users_and_drops_orphan_answer() -> None:
    from dual_mem.memhub.sources import Message

    messages = [
        Message("assistant", "orphan", "s", None, 0, 1),
        Message("user", "a", "s", None, 1, 2),
        Message("user", "b", "s", None, 2, 3),
        Message("assistant", "answer", "s", None, 3, 4),
        Message("user", "open", "s", None, 4, 5),
    ]
    done, open_turn = build_turns(messages)
    assert [(turn.user, turn.assistant) for turn in done] == [("a\nb", "answer")]
    assert open_turn is not None and open_turn.user == "open"


async def test_core_auth_durable_queue_and_single_scope(
    tmp_path: Path,
    fake_embed,
    fake_llm,
    fake_jev,
) -> None:
    core = _memory_client(tmp_path, fake_embed, fake_llm, fake_jev)
    async with _attached_core(tmp_path, core) as (app, remote):
        discovered = resolve_hub_config(tmp_path)
        assert discovered.base_url == "http://127.0.0.1:8765"
        assert discovered.api_key == "secret"
        identity = await remote.identity()
        assert identity["scope"] == {
            "app_id": "default",
            "user_id": HUB_USER_ID,
            "agent_id": "",
        }
        job = await remote.write(
            messages=[
                {"role": "user", "content": "我偏好 PyTorch"},
                {"role": "assistant", "content": "记住了"},
            ],
            session_id="codex:s1",
            request_id="memhub:one",
        )
        assert job["status"] == "done"
        same = await remote.write(
            messages=[
                {"role": "user", "content": "我偏好 PyTorch"},
                {"role": "assistant", "content": "记住了"},
            ],
            session_id="codex:s1",
            request_id="memhub:one",
        )
        assert same["result"]["memory_id"] == job["result"]["memory_id"]
        assert (tmp_path / "hub.db").is_file()
        assert (tmp_path / "uploads").is_dir()

        direct = httpx.AsyncClient(
            base_url="http://hub",
            transport=httpx.ASGITransport(app=app),
        )
        assert (await direct.get("/v1/hub/identity")).status_code == 401
        assert (
            await direct.get(
                "/v1/hub/identity",
                headers={"Authorization": "Bearer wrong"},
            )
        ).status_code == 403
        await direct.aclose()

        recall = await remote.recall(query="PyTorch")
        assert any(
            "PyTorch" in item["content"]
            for group in recall["memories"].values()
            for item in group
        )
        status = await remote.status()
        assert status["memories"]["raw_by_source"] == {"codex": 1}
    assert not (tmp_path / "runtime.json").exists()
    assert oct((tmp_path / "runtime-token").stat().st_mode & 0o777) == "0o600"


async def test_core_serializes_concurrent_writes(tmp_path: Path) -> None:
    class SlowClient:
        def __init__(self) -> None:
            self.settings = Settings(storage_dir=str(tmp_path / "store"))
            self.active = 0
            self.max_active = 0

        async def add(self, **kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.03)
            self.active -= 1
            return SimpleNamespace(
                to_dict=lambda: {
                    "success": True,
                    "memory_id": kwargs["request_id"],
                }
            )

        async def add_raw(self, **kwargs):
            return await self.add(**kwargs)

    slow = SlowClient()
    app = create_hub_app(
        api_key="secret",
        hub_dir=tmp_path,
        client=slow,  # type: ignore[arg-type]
        settings=slow.settings,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with HubClient(
            base_url="http://hub",
            api_key="secret",
            transport=transport,
        ) as remote:
            await asyncio.gather(
                *[
                    remote.write(content=f"memory {i}", request_id=f"r{i}")
                    for i in range(4)
                ]
            )
    assert slow.max_active == 1


async def test_scan_all_hosts_through_core(
    tmp_path: Path,
    fake_embed,
    fake_llm,
    fake_jev,
) -> None:
    hub = tmp_path / "hub"
    roots = _seed_transcripts(tmp_path / "home")
    core = _memory_client(hub, fake_embed, fake_llm, fake_jev)
    async with _attached_core(hub, core) as (_app, remote):
        summary = await scan_sources(
            remote,
            hub_dir=hub,
            roots=roots,
            idle_seconds=0,
        )
        assert {key: value["turns"] for key, value in summary.items()} == dict.fromkeys(
            SOURCE_SPECS,
            1,
        )
        again = await scan_sources(
            remote,
            hub_dir=hub,
            roots=roots,
            idle_seconds=0,
        )
        assert all(value["turns"] == 0 for value in again.values())
        nodes = await core.list(
            user_id=HUB_USER_ID,
            agent_id="",
            limit=100,
            include_history=True,
        )
        assert {(node.layer, node.session_id.partition(":")[0]) for node in nodes if node.layer == Layer.L0_RAW.value} == {
            (Layer.L0_RAW.value, source) for source in SOURCE_SPECS
        }


async def test_open_turn_cursor_and_raw_idempotency(
    tmp_path: Path,
    fake_embed,
    fake_llm,
    fake_jev,
) -> None:
    core = _memory_client(tmp_path, fake_embed, fake_llm, fake_jev)
    path = tmp_path / "ag.jsonl"
    _jsonl(
        path,
        [
            {"type": "user", "session_id": "s", "content": "第一问"},
            {"type": "assistant", "session_id": "s", "content": "第一答"},
            {"type": "user", "session_id": "s", "content": "第二问"},
            {"type": "assistant", "session_id": "s", "content": "中间步骤"},
        ],
    )
    async with _attached_core(tmp_path, core) as (_app, remote):
        with CursorStore(tmp_path) as cursors:
            kwargs = {
                "client": remote,
                "source": "agentica",
                "path": path,
                "cursors": cursors,
            }
            assert await scan_file(idle_seconds=3600, raw_only=True, **kwargs) == 1
            _jsonl(
                path,
                [{"type": "assistant", "session_id": "s", "content": "最终答案"}],
                mode="a",
            )
            assert await scan_file(complete=True, raw_only=True, **kwargs) == 1
            assert await scan_file(complete=True, raw_only=True, **kwargs) == 0
        first = await core.add_raw(
            content="固定原文",
            user_id=HUB_USER_ID,
            request_id="raw-idempotent",
        )
        second = await core.add_raw(
            content="固定原文",
            user_id=HUB_USER_ID,
            request_id="raw-idempotent",
        )
        assert first.memory_id == second.memory_id


def test_install_uninstall_hosts_use_remote_proxy(tmp_path: Path, monkeypatch) -> None:
    from agentica.shell_hooks.config import load_shell_hooks_config

    hub = tmp_path / "hub"
    save_hub_config(
        HubConfig(base_url="http://127.0.0.1:8765", api_key="secret"),
        hub,
    )
    home = tmp_path / "home"
    (home / ".agentica").mkdir(parents=True)
    (home / ".agentica" / "mcp_config.json").write_text(
        json.dumps({"mcpServers": {"other": {"command": "x"}}})
    )
    (home / ".agentica" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "settings": {
                    "hooks": {
                        "consumers": [
                            {"name": "open-island", "command": ["/bin/oi"]}
                        ]
                    }
                }
            }
        )
    )
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text('model = "gpt-5"\n')
    (home / ".claude").mkdir()
    (home / ".claude" / "settings.json").write_text("{}")

    for host in HOSTS:
        install_host(host, hub_dir=hub, home=home)
    status = hosts_status(home=home)
    assert all(
        item["skill"] and item["mcp"] and item["hooks"] in (True, None)
        for item in status.values()
    )
    mcp = json.loads(
        (home / ".agentica" / "mcp_config.json").read_text()
    )["mcpServers"]["dual-mem"]
    assert mcp["args"][:2] == ["-m", "dual_mem.memhub.mcp"]
    assert "env" not in mcp and "secret" not in json.dumps(mcp)
    skill = (
        home
        / ".claude"
        / "skills"
        / "dual-mem-recall"
        / "SKILL.md"
    ).read_text()
    assert "不传 `user_id`" in skill

    for key in ("ENABLED", "CONSUMERS"):
        monkeypatch.delenv(f"AGENTICA_HOOKS_{key}", raising=False)
    hooks = load_shell_hooks_config(
        yaml.safe_load((home / ".agentica" / "config.yaml").read_text())
    )
    assert [item.name for item in hooks.consumers_for("run.completed")] == [
        "open-island",
        "dual-mem-hub",
    ]
    for host in HOSTS:
        uninstall_host(host, home=home)
    assert not any(
        value
        for item in hosts_status(home=home).values()
        for value in item.values()
    )


def test_cli_attach_capture_hook_and_status(tmp_path: Path, monkeypatch) -> None:
    class FakeAttach:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            self.base_url = base_url
            self.api_key = api_key

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def identity(self):
            return {"instance_id": "core-1"}

    monkeypatch.setattr(cli_hub, "HubClient", FakeAttach)
    hub = tmp_path / "hub"
    result = runner.invoke(
        cli_main.app,
        [
            "hub",
            "attach",
            "--base-url",
            "http://127.0.0.1:8765",
            "--api-key",
            "secret",
            "--hub-dir",
            str(hub),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["config"]["api_key"] == "***"
    assert oct((hub / "hub.json").stat().st_mode & 0o777) == "0o600"

    class FakeRemote:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

        async def recall(self, **_kwargs):
            return {
                "memories": {
                    "profile": [],
                    "proactive": [],
                    "normal": [
                        {
                            "memory_id": "m1",
                            "content": "我偏好 PyTorch",
                            "category": "fact",
                            "score": 1.0,
                            "tags": [],
                            "session_id": "codex:s1",
                            "layer": "L1_FACT",
                            "kind": "fact",
                            "is_latest": True,
                            "revision": 0,
                            "source_ids": [],
                            "supporting_ids": [],
                            "supersedes": [],
                            "conflicts_with": [],
                        }
                    ],
                    "raw": [],
                    "schema": [],
                    "catalog": [],
                },
                "conversations": [],
            }

        async def status(self):
            return {"instance_id": "core-1", "memories": {}, "writes": {}}

    monkeypatch.setattr(cli_hub, "make_client", lambda _hub: FakeRemote())
    hook = runner.invoke(
        cli_main.app,
        ["hub", "hook-search", "--hub-dir", str(hub)],
        input=json.dumps({"prompt": "我的框架偏好"}, ensure_ascii=False),
    )
    assert hook.exit_code == 0 and "PyTorch" in hook.output
    status = runner.invoke(
        cli_main.app,
        ["hub", "status", "--hub-dir", str(hub)],
    )
    assert json.loads(status.output)["instance_id"] == "core-1"

    spawned = []
    monkeypatch.setattr(
        cli_hub,
        "_spawn_detached",
        lambda argv, _log: spawned.append(argv),
    )
    transcript = tmp_path / "turn.jsonl"
    capture = runner.invoke(
        cli_main.app,
        ["hub", "capture", "claude-code", "--hub-dir", str(hub)],
        input=json.dumps({"transcript_path": str(transcript)}),
    )
    assert capture.exit_code == 0
    assert spawned == [
        [
            sys.executable,
            "-m",
            "dual_mem.cli.main",
            "hub",
            "scan",
            "--source",
            "claude-code",
            "--file",
            str(transcript),
            "--complete",
            "--hub-dir",
            str(hub),
        ]
    ]
    detached = runner.invoke(
        cli_main.app,
        ["hub", "detach", "--hub-dir", str(hub)],
    )
    assert json.loads(detached.output)["detached"] is True
    assert not (hub / "hub.json").exists()


def test_cli_attach_rejects_plaintext_non_loopback(tmp_path: Path) -> None:
    result = runner.invoke(
        cli_main.app,
        [
            "hub",
            "attach",
            "--base-url",
            "http://memory.example.com",
            "--api-key",
            "secret",
            "--hub-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "require HTTPS" in result.output
