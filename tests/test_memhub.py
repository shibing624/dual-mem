# -*- coding: utf-8 -*-
"""MemHub Core, transcript intake and host wiring tests."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
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
from dual_mem.memhub.queue import HubWriteQueue
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


async def _drain(app) -> None:
    """Wait for the Core's writer to finish the jobs already accepted."""
    while app.state.queue.counts().get("pending", 0) or app.state.queue.counts().get(
        "running", 0
    ):
        await asyncio.sleep(0.01)


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
            wait=True,
        )
        assert job["status"] == "done"
        same = await remote.write(
            messages=[
                {"role": "user", "content": "我偏好 PyTorch"},
                {"role": "assistant", "content": "记住了"},
            ],
            session_id="codex:s1",
            request_id="memhub:one",
            wait=True,
        )
        assert same["result"]["memory_id"] == job["result"]["memory_id"]
        assert (tmp_path / "hub.db").is_file()

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
                    remote.write(content=f"memory {i}", request_id=f"r{i}", wait=True)
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
        await _drain(_app)
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


def test_agentica_yaml_install_keeps_comments_and_uninstall_restores_bytes(
    tmp_path: Path,
) -> None:
    hub = tmp_path / "hub"
    save_hub_config(HubConfig(base_url="http://127.0.0.1:8765", api_key="k"), hub)
    home = tmp_path / "home"
    (home / ".agentica").mkdir(parents=True)
    original = (
        "# Agentica configuration - hand-edited.\n"
        "profiles:\n"
        "  main:\n"
        "    model_name: gpt-5.5  # main model\n"
        "    extra_headers:\n"
        '      Routing: "true"\n'
        "settings:\n"
        "  hooks:\n"
        "    enabled: true\n"
        "    # open-island owns this consumer\n"
        "    consumers:\n"
        "      - name: open-island\n"
        "        command:\n"
        '          - "/Applications/Open Island/bin/hooks"\n'
    )
    config = home / ".agentica" / "config.yaml"
    config.write_text(original)

    install_host("agentica", hub_dir=hub, home=home)
    installed = config.read_text()
    for line in (
        "# Agentica configuration - hand-edited.",
        "model_name: gpt-5.5  # main model",
        'Routing: "true"',
        "# open-island owns this consumer",
        '- "/Applications/Open Island/bin/hooks"',
    ):
        assert line in installed
    assert [c["name"] for c in yaml.safe_load(installed)["settings"]["hooks"]["consumers"]] == [
        "open-island",
        "dual-mem-hub",
    ]

    uninstall_host("agentica", home=home)
    assert config.read_text() == original


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
    assert hook.exit_code == 0
    context = json.loads(hook.output)["hookSpecificOutput"]["additionalContext"]
    assert context.startswith("<relevant-memories>") and "PyTorch" in context
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
            "dual_mem.cli",
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


async def test_failed_write_is_retried_and_cursor_never_skips_it(
    tmp_path: Path, fake_embed, fake_llm, fake_jev
) -> None:
    class FlakyWrite:
        """Model unavailable at first, then healthy."""

        def __init__(self, core: MemoryClient) -> None:
            self.core = core
            self.settings = core.settings
            self.factory = core.factory
            self.down = True
            self.calls = 0

        async def add(self, **kwargs):
            self.calls += 1
            if self.down:
                raise RuntimeError("429 rate limited")
            return await self.core.add(**kwargs)

        async def add_raw(self, **kwargs):
            return await self.core.add_raw(**kwargs)

    core = _memory_client(tmp_path / "hub", fake_embed, fake_llm, fake_jev)
    flaky = FlakyWrite(core)
    hub = tmp_path / "hub"
    app = create_hub_app(
        api_key="secret",
        hub_dir=hub,
        client=flaky,  # type: ignore[arg-type]
        settings=core.settings,
        queue_retry_seconds=0.01,
    )
    path = tmp_path / "ag.jsonl"
    _jsonl(
        path,
        [
            {"type": "user", "session_id": "s", "content": "我喜欢 Rust"},
            {"type": "assistant", "session_id": "s", "content": "好"},
        ],
    )
    async with app.router.lifespan_context(app):
        async with HubClient(
            base_url="http://hub",
            api_key="secret",
            transport=httpx.ASGITransport(app=app),
        ) as remote:
            with CursorStore(hub) as cursors:
                kwargs = {
                    "client": remote,
                    "source": "agentica",
                    "path": path,
                    "cursors": cursors,
                    "complete": True,
                }
                assert await scan_file(**kwargs) == 1
                # The evidence is durable and queued while the model is down.
                assert app.state.queue.counts()["pending"] == 1
                while flaky.calls == 0:
                    await asyncio.sleep(0.01)
                assert app.state.queue.counts().get("done", 0) == 0
                flaky.down = False
                for _ in range(500):
                    if app.state.queue.counts().get("done"):
                        break
                    await asyncio.sleep(0.01)
                assert app.state.queue.counts().get("done") == 1
                assert app.state.queue.counts().get("failed", 0) == 0
                assert flaky.calls >= 2
                assert await scan_file(**kwargs) == 0
            # The retry reused the same L0 node instead of storing a second copy.
            raws = [
                n
                for n in await core.list(user_id=HUB_USER_ID, limit=50, include_history=True)
                if n.layer == Layer.L0_RAW.value
            ]
            assert len(raws) == 1


async def test_write_gives_up_after_max_attempts_then_retry_requeues(
    tmp_path: Path, fake_embed, fake_llm, fake_jev
) -> None:
    calls = []

    async def broken(_payload):
        calls.append(1)
        raise RuntimeError("400 bad model response")

    queue = HubWriteQueue(
        tmp_path / "hub.db", broken, max_attempts=2, retry_base_seconds=0.01
    )
    queue.start()
    queue.submit({"content": "x", "messages": None, "session_id": "", "memory_at": None,
                  "request_id": "r1", "raw_only": False})
    for _ in range(500):
        if queue.get("r1")["status"] == "failed":
            break
        await asyncio.sleep(0.01)
    job = queue.get("r1")
    assert (job["status"], job["attempts"]) == ("failed", 2)
    assert "400 bad model response" in job["error"]
    assert len(calls) == 2

    # Resubmitting the same request puts it back in the queue instead of returning the failure.
    queue.submit({"content": "x", "messages": None, "session_id": "", "memory_at": None,
                  "request_id": "r1", "raw_only": False})
    assert queue.get("r1")["status"] in ("pending", "running")
    await queue.close()


async def test_raw_only_scan_can_be_extracted_later(
    tmp_path: Path, fake_embed, fake_llm, fake_jev
) -> None:
    hub = tmp_path / "hub"
    core = _memory_client(hub, fake_embed, fake_llm, fake_jev)
    async with _attached_core(hub, core) as (app, remote):
        path = tmp_path / "ag.jsonl"
        _jsonl(
            path,
            [
                {"type": "user", "session_id": "s", "content": "我常用 PyTorch 训练"},
                {"type": "assistant", "session_id": "s", "content": "记下了"},
            ],
        )
        with CursorStore(hub) as cursors:
            assert await scan_file(
                remote,
                source="agentica",
                path=path,
                cursors=cursors,
                complete=True,
                raw_only=True,
            ) == 1
        await _drain(app)
        assert app.state.queue.counts()["raw_only_done"] == 1
        assert await core.list(user_id=HUB_USER_ID, limit=50) != []
        raw_ids = {
            node.memory_id
            for node in await core.list(user_id=HUB_USER_ID, limit=50, include_history=True)
            if node.layer == Layer.L0_RAW.value
        }
        assert app.state.queue.counts()["done"] == 1

        assert (await remote.extract_raw())["queued"] == 1
        await _drain(app)
        assert app.state.queue.counts()["raw_only_done"] == 0
        extracted = [
            node
            for node in await core.list(user_id=HUB_USER_ID, limit=50, include_history=True)
            if node.layer != Layer.L0_RAW.value
        ]
        assert extracted, "extraction created facts from the raw evidence"
        # Extraction reused the stored L0 rather than writing a second copy.
        assert {
            node.memory_id
            for node in await core.list(user_id=HUB_USER_ID, limit=50, include_history=True)
            if node.layer == Layer.L0_RAW.value
        } == raw_ids


async def test_core_processes_oldest_event_time_first(tmp_path: Path) -> None:
    order = []

    async def record(payload):
        order.append(payload["content"])
        return {}

    queue = HubWriteQueue(tmp_path / "hub.db", record)

    def submit(content: str, at: int) -> None:
        queue.submit(
            {"content": content, "messages": None, "session_id": "", "memory_at": at,
             "request_id": content, "raw_only": False}
        )

    # Backlog accepted before the writer runs: a live turn, then older transcripts.
    submit("live", 1791000200)
    submit("old", 1791000000)
    submit("older", 1790900000)
    queue.start()
    while queue.counts().get("done", 0) < 3:
        await asyncio.sleep(0.01)
    await queue.close()
    assert order == ["older", "old", "live"]


async def test_recall_skips_the_group_the_caller_disabled(
    tmp_path: Path, fake_embed, fake_llm, fake_jev
) -> None:
    calls = []
    core = _memory_client(tmp_path / "hub", fake_embed, fake_llm, fake_jev)

    class Counted:
        settings = core.settings
        factory = core.factory

        async def search(self, **kwargs):
            calls.append("memories")
            return await core.search(**kwargs)

        async def search_conversation(self, **kwargs):
            calls.append("conversations")
            return await core.search_conversation(**kwargs)

    app = create_hub_app(
        api_key="secret", hub_dir=tmp_path / "hub", client=Counted(), settings=core.settings
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with HubClient(base_url="http://hub", api_key="secret", transport=transport) as remote:
            await remote.recall(query="q", limit=0, raw_limit=3, procedure_limit=0)
            assert calls == ["conversations"]
            calls.clear()
            await remote.recall(query="q", limit=3, raw_limit=0)
            assert calls == ["memories"]


async def test_serve_refuses_non_loopback_and_a_second_core_cannot_steal_the_token(
    tmp_path: Path, monkeypatch, fake_embed, fake_llm, fake_jev
) -> None:
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *_a, **_kw: None)
    refused = runner.invoke(
        cli_main.app,
        ["hub", "serve", "--host", "0.0.0.0", "--hub-dir", str(tmp_path)],
    )
    assert refused.exit_code != 0 and "loopback only" in refused.output

    core = _memory_client(tmp_path, fake_embed, fake_llm, fake_jev)
    running = create_hub_app(api_key="owner-token", hub_dir=tmp_path, client=core)
    async with running.router.lifespan_context(running):
        assert (tmp_path / "runtime-token").read_text().strip() == "owner-token"
        # A second Core (different --api-key) cannot take the lock, and must leave
        # the running Core's token usable.
        second = create_hub_app(api_key="attacker", hub_dir=tmp_path)
        with pytest.raises(RuntimeError, match="already owns"):
            await _start_lifespan(second)
        assert (tmp_path / "runtime-token").read_text().strip() == "owner-token"
        assert resolve_hub_config(tmp_path).api_key == "owner-token"


async def _start_lifespan(app) -> None:
    async with app.router.lifespan_context(app):
        pass


def test_install_service_writes_launchd_plist_or_systemd_unit(
    tmp_path: Path, monkeypatch
) -> None:
    from dual_mem.memhub import service

    hub = tmp_path / "hub"
    monkeypatch.setattr(service, "launchd_plist_path", lambda label: tmp_path / f"{label}.plist")
    monkeypatch.setattr(service, "systemd_unit_path", lambda unit: tmp_path / unit)
    commands = []

    def runner(argv):
        commands.append(argv)
        return {"command": argv, "returncode": 0, "stderr": ""}

    info = service.install_service(hub_dir=hub, port=9999, runner=runner)
    text = Path(info["service_file"]).read_text(errors="replace") if sys.platform != "darwin" else Path(
        info["service_file"]
    ).read_bytes().decode()
    assert "hub" in text and "serve" in text and "9999" in text and str(hub.resolve()) in text
    if sys.platform == "darwin":
        assert commands[0][:1] == ["launchctl"] and commands[-1][1] == "load"
    else:
        assert ["systemctl", "--user", "enable", service.SYSTEMD_UNIT] in commands

    removed = service.uninstall_service(runner=runner)
    assert removed["removed"] is True and not Path(removed["service_file"]).exists()


def test_mcp_builds_without_a_running_core(tmp_path: Path) -> None:
    """A host that starts the MCP server before `hub serve` must not crash at import time."""
    pytest.importorskip("mcp.server.mcpserver")
    from dual_mem.memhub.mcp import build_hub_mcp

    server = build_hub_mcp(hub_dir=tmp_path)  # no runtime.json yet: no connection attempted
    assert server is not None


async def test_unreachable_core_leaves_cursor_in_place(tmp_path: Path) -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Core is down")

    path = tmp_path / "ag.jsonl"
    _jsonl(
        path,
        [
            {"type": "user", "session_id": "s", "content": "q"},
            {"type": "assistant", "session_id": "s", "content": "a"},
        ],
    )
    async with HubClient(
        base_url="http://hub", api_key="k", transport=httpx.MockTransport(refuse)
    ) as remote:
        with CursorStore(tmp_path) as cursors:
            with pytest.raises(httpx.ConnectError):
                await scan_file(
                    remote, source="agentica", path=path, cursors=cursors, complete=True
                )
            assert cursors.get("agentica", str(path.resolve())) is None


# --- coding scene: tool steps, version boundary, idle digest ---------------------


def test_claude_and_codex_tool_steps_become_turn_steps(tmp_path: Path) -> None:
    roots = default_source_roots(tmp_path)
    agentica = roots["agentica"] / "default" / "proj" / "ag-tools.jsonl"
    # Shape of a real agentica session log: tool_calls on the assistant record,
    # one ``tool`` record per result (parallel calls), cwd / git_branch on each.
    _jsonl(
        agentica,
        [
            {"type": "event", "name": "session_meta", "session_id": "ag"},
            {"type": "user", "session_id": "ag", "cwd": "/proj", "git_branch": "fix/slug",
             "content": "slugify does nothing"},
            {"type": "assistant", "session_id": "ag", "cwd": "/proj", "git_branch": "fix/slug",
             "content": "", "tool_calls": [
                 {"id": "a1", "type": "function",
                  "function": {"name": "read_file", "arguments": "{\"file_path\": \"t.py\"}"}},
                 {"id": "a2", "type": "function",
                  "function": {"name": "glob", "arguments": "{\"pattern\": \"AGENTS.md\"}"}},
             ]},
            {"type": "tool", "session_id": "ag", "cwd": "/proj", "git_branch": "fix/slug",
             "tool_call_id": "a1", "tool_name": "read_file", "content": "def slugify(s): return s"},
            {"type": "tool", "session_id": "ag", "cwd": "/proj", "git_branch": "fix/slug",
             "tool_call_id": "a2", "tool_name": "glob", "content": "[]"},
            {"type": "event", "name": "token_usage", "session_id": "ag"},
            {"type": "assistant", "session_id": "ag", "cwd": "/proj", "git_branch": "fix/slug",
             "content": "fixed slugify"},
        ],
    )
    messages, _ = SOURCE_SPECS["agentica"].read(agentica, 0)
    done, open_turn = build_turns(messages)
    turn = (done + [open_turn])[0]
    assert (turn.user, turn.assistant) == ("slugify does nothing", "fixed slugify")
    assert (turn.workspace, turn.branch) == ("/proj", "fix/slug")
    msgs = turn.messages()
    assert [m["role"] for m in msgs] == ["user", "assistant", "assistant", "tool", "tool", "assistant"]
    assert [m["tool_calls"][0]["function"]["name"] for m in msgs[1:3]] == ["read_file", "glob"]
    assert json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]) == {"file_path": "t.py"}
    assert msgs[3] == {"role": "tool", "tool_call_id": "a1", "content": "def slugify(s): return s"}
    assert msgs[4] == {"role": "tool", "tool_call_id": "a2", "content": "[]"}

    claude = roots["claude-code"] / "proj" / "cc-tools.jsonl"
    _jsonl(
        claude,
        [
            {"type": "user", "sessionId": "cc", "cwd": "/repo", "gitBranch": "main",
             "message": {"content": "fix the flaky test"}},
            {"type": "assistant", "sessionId": "cc", "cwd": "/repo", "gitBranch": "main",
             "message": {"content": [
                 {"type": "text", "text": "running it"},
                 {"type": "tool_use", "id": "t1", "name": "Bash",
                  "input": {"command": "pytest -q tests/test_x.py"}},
             ]}},
            {"type": "user", "sessionId": "cc", "cwd": "/repo", "gitBranch": "fix",
             "message": {"content": [
                 {"type": "tool_result", "tool_use_id": "t1",
                  "content": [{"type": "text", "text": "1 failed: race in setup"}]},
             ]}},
            {"type": "assistant", "sessionId": "cc", "cwd": "/repo", "gitBranch": "fix",
             "message": {"content": [{"type": "text", "text": "fixed the race"}]}},
            {"type": "user", "sessionId": "cc", "message": {"content": "thanks"}},
        ],
    )
    messages, _ = SOURCE_SPECS["claude-code"].read(claude, 0)
    done, open_turn = build_turns(messages)
    (turn,) = done
    assert (turn.user, turn.assistant) == ("fix the flaky test", "fixed the race")
    assert (turn.workspace, turn.branch) == ("/repo", "fix")
    msgs = turn.messages()
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    call = msgs[1]["tool_calls"][0]
    assert call["function"]["name"] == "Bash"
    assert json.loads(call["function"]["arguments"]) == {"command": "pytest -q tests/test_x.py"}
    assert msgs[2] == {"role": "tool", "tool_call_id": "t1", "content": "1 failed: race in setup"}
    assert open_turn is not None and open_turn.user == "thanks"

    codex = roots["codex"] / "2026" / "10" / "04" / "rollout-tools.jsonl"
    _jsonl(
        codex,
        [
            {"type": "session_meta",
             "payload": {"id": "cx", "cwd": "/svc", "git": {"branch": "dev"}}},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {
                "type": "UserMessage", "content": [{"type": "text", "text": "why 500?"}]}}},
            {"type": "response_item", "payload": {
                "type": "function_call", "call_id": "c1", "name": "shell",
                "arguments": "{\"cmd\": \"tail app.log\"}"}},
            {"type": "response_item", "payload": {
                "type": "function_call_output", "call_id": "c1", "output": "x" * 9000}},
            {"type": "response_item", "payload": {"type": "reasoning", "summary": []}},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {
                "type": "AgentMessage", "content": [{"type": "Text", "text": "DB pool exhausted"}]}}},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {
                "type": "UserMessage", "content": [{"type": "text", "text": "ok"}]}}},
        ],
    )
    messages, _ = SOURCE_SPECS["codex"].read(codex, 0)
    (turn,), _ = build_turns(messages)
    assert (turn.session_id, turn.workspace, turn.branch) == ("cx", "/svc", "dev")
    roles = [m["role"] for m in turn.messages()]
    assert roles == ["user", "assistant", "tool", "assistant"]
    output = turn.messages()[2]["content"]
    assert len(output) < 9000 and output.startswith("x") and output.endswith("x")


async def test_scan_forwards_tool_steps_and_boundary(tmp_path: Path) -> None:
    roots = default_source_roots(tmp_path)
    path = roots["claude-code"] / "proj" / "cc-b.jsonl"
    _jsonl(
        path,
        [
            {"type": "user", "sessionId": "s", "cwd": "/w", "gitBranch": "b",
             "message": {"content": "q"}},
            {"type": "assistant", "sessionId": "s", "message": {"content": [
                {"type": "tool_use", "id": "t", "name": "Read", "input": {"path": "a.py"}}]}},
            {"type": "user", "sessionId": "s", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t", "content": "body"}]}},
            {"type": "assistant", "sessionId": "s", "message": {"content": "a"}},
            {"type": "user", "sessionId": "s", "message": {"content": "next"}},
        ],
    )
    sent: list[dict] = []

    class Writer:
        async def write(self, **kwargs):
            sent.append(kwargs)
            return {"status": "pending"}

    await scan_file(Writer(), source="claude-code", path=path,
                    cursors=CursorStore(tmp_path / "cursors.db"))
    (call,) = sent
    assert (call["workspace_id"], call["branch"]) == ("/w", "b")
    assert [m["role"] for m in call["messages"]] == ["user", "assistant", "tool", "assistant"]


async def _until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0.01)


async def test_idle_hook_runs_after_work_and_survives_failures(tmp_path: Path) -> None:
    runs: list[int] = []
    fail = {"on": True}

    async def process(payload):
        return {}

    async def idle():
        runs.append(1)
        if fail["on"]:
            raise RuntimeError("model down")

    queue = HubWriteQueue(tmp_path / "hub.db", process, on_idle=idle, idle_seconds=0.05)
    queue.start()
    # Once after start (work left by a previous process), failure recorded.
    await asyncio.wait_for(_until(lambda: runs), 5)
    await asyncio.sleep(0.05)
    assert queue.last_idle_error == "RuntimeError: model down"
    await asyncio.sleep(0.15)
    assert len(runs) == 1  # nothing new written: no repeat
    fail["on"] = False
    queue.submit({"content": "x", "messages": None, "session_id": "", "memory_at": None,
                  "request_id": "r", "raw_only": False})
    await asyncio.wait_for(_until(lambda: len(runs) >= 2), 5)
    await asyncio.sleep(0.02)
    assert queue.last_idle_error is None
    await queue.close()


async def test_dual_core_digests_on_idle_and_recalls_procedures(
    tmp_path: Path, fake_embed
) -> None:
    from conftest import FakeJev, FakeLLMClient

    candidate = {"task": "How to fix the flaky race in setup",
                 "solution": "Serialize fixture setup with a lock", "search_keys": []}
    core = MemoryClient(
        settings=Settings(storage_dir=str(tmp_path / "hub" / "store"), mode="dual"),
        embed=fake_embed,
        llm=FakeLLMClient({"json": {"candidates": [candidate]}}),
        jev=FakeJev({"scene": "CODING"}),
    )
    digests: list[int] = []
    original = core.digest

    async def counted():
        digests.append(1)
        return await original()

    core.digest = counted
    app = create_hub_app(api_key="secret", hub_dir=tmp_path / "hub", client=core,
                         settings=core.settings, digest_idle_seconds=0.05)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with HubClient(base_url="http://hub", api_key="secret",
                             transport=transport) as remote:
            receipt = await remote.write(
                messages=[
                    {"role": "user", "content": "fix the flaky test"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "t", "type": "function",
                         "function": {"name": "Bash", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "t", "content": "race in setup"},
                    {"role": "assistant", "content": "fixed"},
                ],
                request_id="r1", wait=True, workspace_id="/repo", branch="main",
            )
            assert receipt["status"] == "done"
            assert receipt["result"]["scene"] == "coding"
            proc = core.factory.vector.get(receipt["result"]["procedure_ids"][0])
            assert proc.custom["procedure"]["workspace_id"] == "/repo"
            recalled = await remote.recall(query=candidate["task"], limit=0, raw_limit=0)
            assert [m["memory_id"] for m in recalled["memories"]["procedures"]] == [
                proc.node_id
            ]
            # The write arrived inside the first idle window: one digest after it.
            await asyncio.wait_for(_until(lambda: digests), 5)
            assert core.factory.cache.list_pending_s2_scopes() == []
            status = await remote.status()
            assert status["digest"]["auto"] is True and status["digest"]["last_error"] is None
    await core.aclose()


async def test_system1_core_never_digests(tmp_path: Path, fake_embed, fake_llm, fake_jev) -> None:
    core = _memory_client(tmp_path / "hub", fake_embed, fake_llm, fake_jev)
    app = create_hub_app(api_key="secret", hub_dir=tmp_path / "hub", client=core,
                         settings=core.settings, digest_idle_seconds=0.01)
    async with app.router.lifespan_context(app):
        assert app.state.queue._on_idle is None
    await core.aclose()


def test_claude_code_hook_keeps_tool_trajectory() -> None:
    from dual_mem.integrations.claude_code import to_messages

    out = to_messages([
        {"type": "user", "message": {"role": "user", "content": "fix it"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "t", "name": "Read", "input": {"path": "a.py"}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": "print(1)"}]}},
        {"role": "assistant", "content": "done"},
        {"type": "summary", "summary": "ignored"},
    ])
    assert out == [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "checking", "tool_calls": [
            {"id": "t", "type": "function",
             "function": {"name": "Read", "arguments": "{\"path\": \"a.py\"}"}}]},
        {"role": "tool", "tool_call_id": "t", "content": "print(1)"},
        {"role": "assistant", "content": "done"},
    ]


def test_connect_reads_remote_core_builds_tunnel_and_attaches(
    tmp_path: Path, monkeypatch
) -> None:
    from dual_mem.memhub import service

    ssh_calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        ssh_calls.append(argv)
        return SimpleNamespace(
            returncode=0,
            stdout='tok-123\n{"base_url": "http://127.0.0.1:8765", "pid": 1}\n',
            stderr="",
        )

    tunnels: list[dict] = []
    monkeypatch.setattr(cli_hub.subprocess, "run", fake_run)
    monkeypatch.setattr(
        service, "install_tunnel",
        lambda target, **kw: tunnels.append({"target": target, **kw})
        or {"service_file": "/x/tunnel.plist", "log": "/x/tunnel.log", "actions": []},
    )
    uninstalled: list[int] = []
    monkeypatch.setattr(service, "uninstall_tunnel", lambda: uninstalled.append(1) or {})
    seen: list[tuple[str, str]] = []

    async def identity(base_url, api_key, *, timeout):
        seen.append((base_url, api_key))
        return {"instance_id": "remote-core"}

    monkeypatch.setattr(cli_hub, "_wait_identity", identity)
    hub = tmp_path / "hub"
    result = runner.invoke(
        cli_main.app,
        ["hub", "connect", "me@box", "--local-port", "18766",
         "--ssh-arg=-p", "--ssh-arg=2222", "--hub-dir", str(hub)],
    )
    assert result.exit_code == 0, result.output
    (argv,) = ssh_calls
    assert argv[:3] == ["ssh", "-o", "BatchMode=yes"] and argv[3:6] == ["-p", "2222", "me@box"]
    assert argv[-1] == "cat ~/.dual-mem/runtime-token && cat ~/.dual-mem/runtime.json"
    assert tunnels == [{"target": "me@box", "local_port": 18766, "remote_port": 8765,
                        "ssh_args": ["-p", "2222"], "hub_dir": str(hub)}]
    assert seen == [("http://127.0.0.1:18766", "tok-123")]
    saved = json.loads((hub / "hub.json").read_text())
    assert saved == {"base_url": "http://127.0.0.1:18766", "api_key": "tok-123",
                     "ssh_target": "me@box"}
    out = json.loads(result.output)
    assert out["config"]["api_key"] == "***" and out["identity"]["instance_id"] == "remote-core"
    assert resolve_hub_config(hub).resolved_base_url() == "http://127.0.0.1:18766"

    detached = runner.invoke(cli_main.app, ["hub", "detach", "--hub-dir", str(hub)])
    assert detached.exit_code == 0 and uninstalled == [1]
    assert not (hub / "hub.json").exists()


def test_connect_fails_loudly_and_removes_a_dead_tunnel(tmp_path: Path, monkeypatch) -> None:
    from dual_mem.memhub import service

    monkeypatch.setattr(
        cli_hub.subprocess, "run",
        lambda argv, **kw: SimpleNamespace(
            returncode=0, stdout='t\n{"base_url": "http://127.0.0.1:9000"}', stderr=""
        ),
    )
    monkeypatch.setattr(
        service, "install_tunnel",
        lambda target, **kw: {"service_file": "f", "log": "/x/tunnel.log", "actions": []},
    )
    removed: list[int] = []
    monkeypatch.setattr(service, "uninstall_tunnel", lambda: removed.append(1) or {})

    async def unreachable(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(cli_hub, "_wait_identity", unreachable)
    hub = tmp_path / "hub"
    result = runner.invoke(cli_main.app, ["hub", "connect", "box", "--hub-dir", str(hub)])
    assert result.exit_code != 0 and "/x/tunnel.log" in result.output
    assert removed == [1] and not (hub / "hub.json").exists()

    monkeypatch.setattr(
        cli_hub.subprocess, "run",
        lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="No such file"),
    )
    result = runner.invoke(cli_main.app, ["hub", "connect", "box", "--hub-dir", str(hub)])
    assert result.exit_code != 0 and "install-service" in result.output


async def test_wait_identity_retries_transport_but_not_a_rejected_token() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) < 3:
            raise httpx.ConnectError("tunnel starting", request=request)
        return httpx.Response(401, json={"detail": "bad token"})

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            cli_hub,
            "HubClient",
            lambda **kw: HubClient(**kw, transport=httpx.MockTransport(handler)),
        )
        real_sleep = asyncio.sleep
        mp.setattr(cli_hub.asyncio, "sleep", lambda _s: real_sleep(0))
        start = time.monotonic()
        with pytest.raises(httpx.HTTPStatusError):
            await cli_hub._wait_identity("http://127.0.0.1:1", "t", timeout=30)
    # Two refused connects are retried; the 401 ends the wait at once, not at 30s.
    assert len(calls) == 3 and time.monotonic() - start < 5


def test_tunnel_service_restarts_always_and_never_prompts(
    tmp_path: Path, monkeypatch
) -> None:
    import plistlib

    from dual_mem.memhub import service

    monkeypatch.setattr(
        service, "launchd_plist_path", lambda label=service.LABEL: tmp_path / f"{label}.plist"
    )
    monkeypatch.setattr(
        service, "systemd_unit_path", lambda unit=service.SYSTEMD_UNIT: tmp_path / unit
    )
    commands: list[list[str]] = []

    def run(argv):
        commands.append(argv)
        return {"command": argv, "returncode": 0, "stderr": ""}

    info = service.install_tunnel(
        "me@box", local_port=18766, remote_port=8765, ssh_args=["-p", "2222"],
        hub_dir=tmp_path / "hub", runner=run,
    )
    argv = service.tunnel_argv("me@box", local_port=18766, remote_port=8765,
                               ssh_args=["-p", "2222"])
    assert argv[:2] == ["ssh", "-N"] and argv[-1] == "me@box"
    assert "BatchMode=yes" in argv and "ExitOnForwardFailure=yes" in argv
    assert argv[argv.index("-L") + 1] == "127.0.0.1:18766:127.0.0.1:8765"
    path = Path(info["service_file"])
    if sys.platform == "darwin":
        plist = plistlib.loads(path.read_bytes())
        assert plist["Label"] == service.TUNNEL_LABEL
        assert plist["ProgramArguments"] == argv and plist["KeepAlive"] is True
    else:
        text = path.read_text()
        assert "Restart=always" in text and "127.0.0.1:18766:127.0.0.1:8765" in text
    # The Core service is a different file: connecting never touches it.
    assert not service.launchd_plist_path(service.LABEL).exists()
    assert service.uninstall_tunnel(runner=run)["removed"] is True
    assert not path.exists()


def test_hook_install_replaces_hooks_written_as_cli_main(tmp_path: Path) -> None:
    """Hooks installed before ``-m dual_mem.cli`` are still recognised as ours."""
    from dual_mem.memhub.hosts import ClaudeStyleHooks, Launch

    hook_file = tmp_path / "settings.json"
    old = f"{sys.executable} -m dual_mem.cli.main hub capture claude-code"
    hook_file.write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {"hooks": [{"type": "command", "command": old}]},
                        {"hooks": [{"type": "command", "command": "other-tool"}]},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    ClaudeStyleHooks().install(hook_file, Launch("claude-code", tmp_path / "hub"))

    stop = json.loads(hook_file.read_text(encoding="utf-8"))["hooks"]["Stop"]
    commands = [e["hooks"][0]["command"] for e in stop]
    assert "other-tool" in commands
    assert old not in commands
    ours = [c for c in commands if "-m dual_mem.cli" in c]
    assert len(ours) == 1 and "-m dual_mem.cli hub" in ours[0]
