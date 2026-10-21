# -*- coding: utf-8 -*-
"""Run the MemHub Core as a login service, so hooks and MCP always find it.

macOS uses a LaunchAgent, Linux a systemd user unit. Both start at login and
restart on failure. No credentials are copied into the service file: like any
dual-mem process, the Core reads model keys from ``~/.dual_mem/config.yaml``
(a login service does not see variables exported in a shell profile).
"""

from __future__ import annotations

import os
import plistlib
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dual_mem.memhub.config import hub_root

LABEL = "com.dual-mem.hub"
SYSTEMD_UNIT = "dual-mem-hub.service"


def _argv(hub_dir: Path, host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dual_mem.cli.main",
        "hub",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
        "--hub-dir",
        str(hub_dir),
    ]


def launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def systemd_unit_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "systemd" / "user" / SYSTEMD_UNIT


def render_launchd_plist(
    *, hub_dir: Path, host: str = "127.0.0.1", port: int = 8765, log_dir: Path | None = None
) -> bytes:
    logs = log_dir or hub_dir / "logs"
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": _argv(hub_dir, host, port),
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "WorkingDirectory": str(hub_dir),
            "StandardOutPath": str(logs / "serve.log"),
            "StandardErrorPath": str(logs / "serve.log"),
        }
    )


def render_systemd_unit(
    *, hub_dir: Path, host: str = "127.0.0.1", port: int = 8765, log_dir: Path | None = None
) -> str:
    logs = log_dir or hub_dir / "logs"
    return "\n".join(
        [
            "[Unit]",
            "Description=dual-mem MemHub Core",
            "After=network.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={hub_dir}",
            f"ExecStart={shlex.join(_argv(hub_dir, host, port))}",
            "Restart=on-failure",
            "RestartSec=5",
            f"StandardOutput=append:{logs / 'serve.log'}",
            f"StandardError=append:{logs / 'serve.log'}",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def _run(argv: list[str]) -> dict[str, Any]:
    result = subprocess.run(argv, capture_output=True, text=True)
    return {"command": argv, "returncode": result.returncode, "stderr": result.stderr.strip()}


def install_service(
    *,
    hub_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    runner: Callable[[list[str]], dict[str, Any]] = _run,
) -> dict[str, Any]:
    """Write and load the login service for this Hub directory."""
    root = hub_root(hub_dir)
    (root / "logs").mkdir(exist_ok=True)
    if sys.platform == "darwin":
        path = launchd_plist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(render_launchd_plist(hub_dir=root, host=host, port=port))
        actions = [runner(["launchctl", "unload", str(path)]), runner(["launchctl", "load", str(path)])]
    else:
        path = systemd_unit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_systemd_unit(hub_dir=root, host=host, port=port), encoding="utf-8")
        actions = [
            runner(["systemctl", "--user", "daemon-reload"]),
            runner(["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT]),
        ]
    return {"service_file": str(path), "actions": actions}


def uninstall_service(*, runner: Callable[[list[str]], dict[str, Any]] = _run) -> dict[str, Any]:
    if sys.platform == "darwin":
        path = launchd_plist_path()
        actions = [runner(["launchctl", "unload", str(path)])] if path.is_file() else []
    else:
        path = systemd_unit_path()
        actions = (
            [runner(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT])]
            if path.is_file()
            else []
        )
    removed = path.is_file()
    if removed:
        path.unlink()
    return {"service_file": str(path), "removed": removed, "actions": actions}
