# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Wire MemHub into each CLI host: the dual-mem MCP server, a recall
skill, and (where the host has hooks) transcript capture / prompt-time recall.

Every host file is edited in place, keeping all other entries; the first edit
leaves ``<file>.dual-mem-hub.bak``. YAML hosts lose comments on rewrite, which is
why the backup exists.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from dual_mem.memhub.config import hub_root

MCP_NAME = "dual-mem"
SKILL_NAME = "dual-mem-recall"
SKILL_MARKER = "<!-- managed by dual-mem hub -->"
AGENTICA_CONSUMER = "dual-mem-hub"
TOML_BEGIN = "# >>> dual-mem hub >>>"
TOML_END = "# <<< dual-mem hub <<<"
# Our hook commands always run this module, which is how we find them again.
HOOK_SIGNATURE = "-m dual_mem.cli.main hub "
# agentica.shell_hooks.config.SHELL_HOOK_EVENTS; unlisted events default to on there.
AGENTICA_EVENTS = (
    "run.started",
    "run.completed",
    "run.failed",
    "run.cancelled",
    "tool.started",
    "tool.completed",
    "session.started",
    "session.ended",
    "needs.approval",
    "needs.input",
    "needs.resolved",
)
AGENTICA_CAPTURE_EVENTS = ("run.completed", "session.ended")


@dataclass(frozen=True)
class Launch:
    """Host commands use this interpreter and read one 0600 Hub attachment."""

    host: str
    hub_dir: Path

    def dual_mem(self, *args: str) -> list[str]:
        return [sys.executable, "-m", "dual_mem.cli.main", *args]

    def mcp(self) -> dict[str, Any]:
        return {
            "command": sys.executable,
            "args": [
                "-m",
                "dual_mem.memhub.mcp",
                "--hub-dir",
                str(self.hub_dir),
            ],
        }

    def capture(self) -> list[str]:
        return self.dual_mem("hub", "capture", self.host, "--hub-dir", str(self.hub_dir))

    def recall_hook(self) -> list[str]:
        return self.dual_mem("hub", "hook-search", "--hub-dir", str(self.hub_dir))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_name(path.name + ".dual-mem-hub.bak")
    if path.is_file() and not backup.exists():
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".dual-mem-hub.tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.is_file():
        shutil.copymode(path, tmp)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8") or "{}") if path.is_file() else {}


def _dump_json(path: Path, data: dict) -> None:
    _write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _load_yaml(path: Path) -> dict:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}


def _dump_yaml(path: Path, data: dict) -> None:
    _write(path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


class Part(Protocol):
    def install(self, path: Path, launch: Launch) -> None: ...
    def uninstall(self, path: Path) -> bool: ...
    def installed(self, path: Path) -> bool: ...


class JsonMcp:
    """``{"mcpServers": {...}}`` — Agentica, Claude Code (~/.claude.json), Cursor."""

    def install(self, path: Path, launch: Launch) -> None:
        data = _load_json(path)
        data.setdefault("mcpServers", {})[MCP_NAME] = launch.mcp()
        _dump_json(path, data)

    def uninstall(self, path: Path) -> bool:
        data = _load_json(path)
        if MCP_NAME not in data.get("mcpServers", {}):
            return False
        del data["mcpServers"][MCP_NAME]
        _dump_json(path, data)
        return True

    def installed(self, path: Path) -> bool:
        return MCP_NAME in _load_json(path).get("mcpServers", {})


class YamlMcp:
    """Hermes ``config.yaml`` ``mcp_servers``."""

    def install(self, path: Path, launch: Launch) -> None:
        data = _load_yaml(path)
        data.setdefault("mcp_servers", {})[MCP_NAME] = launch.mcp()
        _dump_yaml(path, data)

    def uninstall(self, path: Path) -> bool:
        data = _load_yaml(path)
        if MCP_NAME not in (data.get("mcp_servers") or {}):
            return False
        del data["mcp_servers"][MCP_NAME]
        _dump_yaml(path, data)
        return True

    def installed(self, path: Path) -> bool:
        return MCP_NAME in (_load_yaml(path).get("mcp_servers") or {})


class TomlMcp:
    """Codex ``config.toml``: a marked block, so the rest of the file stays byte-identical."""

    _block = re.compile(rf"\n?{re.escape(TOML_BEGIN)}.*?{re.escape(TOML_END)}\n", re.DOTALL)

    def install(self, path: Path, launch: Launch) -> None:
        text = self._block.sub("", path.read_text(encoding="utf-8")) if path.is_file() else ""
        mcp = launch.mcp()
        block = (
            f"{TOML_BEGIN}\n[mcp_servers.{MCP_NAME}]\n"
            f"command = {json.dumps(mcp['command'])}\n"
            f"args = {json.dumps(mcp['args'])}\n{TOML_END}\n"
        )
        sep = "\n" if text and not text.endswith("\n") else ""
        _write(path, f"{text}{sep}\n{block}" if text else block)

    def uninstall(self, path: Path) -> bool:
        if not self.installed(path):
            return False
        _write(path, self._block.sub("", path.read_text(encoding="utf-8")))
        return True

    def installed(self, path: Path) -> bool:
        return path.is_file() and TOML_BEGIN in path.read_text(encoding="utf-8")


class AgenticaHooks:
    """``settings.hooks.consumers`` in ~/.agentica/config.yaml (fire-and-forget notices)."""

    def install(self, path: Path, launch: Launch) -> None:
        data = _load_yaml(path)
        hooks = data.setdefault("settings", {}).setdefault("hooks", {})
        hooks["enabled"] = True
        consumers = [
            c for c in hooks.get("consumers") or [] if c.get("name") != AGENTICA_CONSUMER
        ]
        consumers.append(
            {
                "name": AGENTICA_CONSUMER,
                "command": launch.capture(),
                "events": {e: e in AGENTICA_CAPTURE_EVENTS for e in AGENTICA_EVENTS},
            }
        )
        hooks["consumers"] = consumers
        _dump_yaml(path, data)

    def uninstall(self, path: Path) -> bool:
        data = _load_yaml(path)
        hooks = (data.get("settings") or {}).get("hooks") or {}
        consumers = hooks.get("consumers") or []
        kept = [c for c in consumers if c.get("name") != AGENTICA_CONSUMER]
        if len(kept) == len(consumers):
            return False
        hooks["consumers"] = kept
        _dump_yaml(path, data)
        return True

    def installed(self, path: Path) -> bool:
        hooks = (_load_yaml(path).get("settings") or {}).get("hooks") or {}
        return any(c.get("name") == AGENTICA_CONSUMER for c in hooks.get("consumers") or [])


class ClaudeStyleHooks:
    """Claude Code ``settings.json`` / Codex ``hooks.json``: recall on prompt, capture on Stop."""

    @staticmethod
    def _ours(entry: dict) -> bool:
        return any(HOOK_SIGNATURE in h.get("command", "") for h in entry.get("hooks", []))

    def install(self, path: Path, launch: Launch) -> None:
        data = _load_json(path)
        hooks = data.setdefault("hooks", {})
        for event, argv in (("UserPromptSubmit", launch.recall_hook()), ("Stop", launch.capture())):
            entries = [e for e in hooks.get(event, []) if not self._ours(e)]
            entries.append({"hooks": [{"type": "command", "command": shlex.join(argv)}]})
            hooks[event] = entries
        _dump_json(path, data)

    def uninstall(self, path: Path) -> bool:
        data = _load_json(path)
        hooks = data.get("hooks", {})
        changed = False
        for event, entries in list(hooks.items()):
            kept = [e for e in entries if not self._ours(e)]
            if len(kept) != len(entries):
                hooks[event] = kept
                changed = True
        if changed:
            _dump_json(path, data)
        return changed

    def installed(self, path: Path) -> bool:
        hooks = _load_json(path).get("hooks", {})
        return any(self._ours(e) for entries in hooks.values() for e in entries)


@dataclass(frozen=True)
class HostSpec:
    # Relative to $HOME; the skill lives at <home_dir>/skills/<SKILL_NAME>/SKILL.md.
    home_dir: str
    mcp: tuple[Part, str]
    hooks: tuple[Part, str] | None = None


HOSTS: dict[str, HostSpec] = {
    "agentica": HostSpec(
        ".agentica", (JsonMcp(), ".agentica/mcp_config.json"), (AgenticaHooks(), ".agentica/config.yaml")
    ),
    "claude-code": HostSpec(
        ".claude", (JsonMcp(), ".claude.json"), (ClaudeStyleHooks(), ".claude/settings.json")
    ),
    # Codex runs hooks only with `[features] hooks = true` and after the user trusts them.
    "codex": HostSpec(
        ".codex", (TomlMcp(), ".codex/config.toml"), (ClaudeStyleHooks(), ".codex/hooks.json")
    ),
    "cursor": HostSpec(".cursor", (JsonMcp(), ".cursor/mcp.json")),
    # Hermes is scanned from its state.db; it gets MCP + skill, no hooks.
    "hermes": HostSpec(".hermes", (YamlMcp(), ".hermes/config.yaml")),
}


def render_skill(*, host: str, recall_argv: list[str]) -> str:
    return f"""---
name: {SKILL_NAME}
description: 用户的全局个人记忆（dual-mem MemHub，跨 Agentica / Claude Code / Codex / Cursor / Hermes 共享）。回答涉及用户偏好、兴趣、长期约定、过往决策，或用户问「我之前问过什么 / 之前在别的 CLI 聊到哪了」时使用。
---

{SKILL_MARKER}

# dual-mem MemHub 个人记忆

本机所有 CLI 通过 loopback HTTP 访问同一个 Hub Core，当前宿主是 `{host}`。Core 是唯一写者，
MCP 工具已经绑定个人 scope，不传 `user_id` / `agent_id` / `app_id`。

- 偏好、兴趣、画像、历史结论：`memory_search(query)`。
- 「我之前问了什么 / 原话是什么」：`conversation_search(query)`。结果的 `session_id`
  形如 `codex:<id>`，冒号前就是来源 CLI。
- 用户明确表达新的长期偏好或约定：`memory_add(content)`。普通对话会被自动采集，无需重复写入。

没有加载 MCP 工具时，用命令行同时查记忆与历史原话：

```bash
{shlex.join(recall_argv)} "要查的内容"
```
"""


def _paths(host: str, home: Path) -> tuple[HostSpec, Path]:
    spec = HOSTS[host]
    return spec, home / spec.home_dir / "skills" / SKILL_NAME / "SKILL.md"


def install_host(
    host: str, *, hub_dir: str | Path | None = None, home: Path | None = None
) -> dict[str, Any]:
    home = home or Path.home()
    spec, skill = _paths(host, home)
    launch = Launch(host, hub_root(hub_dir))
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(
        render_skill(
            host=host,
            recall_argv=launch.dual_mem("hub", "recall", "--hub-dir", str(launch.hub_dir)),
        ),
        encoding="utf-8",
    )
    written = [str(skill)]
    for part, rel in [spec.mcp, *([spec.hooks] if spec.hooks else [])]:
        part.install(home / rel, launch)
        written.append(str(home / rel))
    return {"host": host, "files": written}


def uninstall_host(host: str, *, home: Path | None = None) -> dict[str, Any]:
    home = home or Path.home()
    spec, skill = _paths(host, home)
    skill_removed = skill.is_file() and SKILL_MARKER in skill.read_text(encoding="utf-8")
    if skill_removed:
        shutil.rmtree(skill.parent)
    mcp_part, mcp_rel = spec.mcp
    return {
        "host": host,
        "skill": skill_removed,
        "mcp": mcp_part.uninstall(home / mcp_rel),
        "hooks": spec.hooks[0].uninstall(home / spec.hooks[1]) if spec.hooks else False,
    }


def hosts_status(*, home: Path | None = None) -> dict[str, dict[str, bool | None]]:
    home = home or Path.home()
    out: dict[str, dict[str, bool | None]] = {}
    for host, spec in HOSTS.items():
        _, skill = _paths(host, home)
        mcp_part, mcp_rel = spec.mcp
        out[host] = {
            "skill": skill.is_file() and SKILL_MARKER in skill.read_text(encoding="utf-8"),
            "mcp": mcp_part.installed(home / mcp_rel),
            "hooks": spec.hooks[0].installed(home / spec.hooks[1]) if spec.hooks else None,
        }
    return out
