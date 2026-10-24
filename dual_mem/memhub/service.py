# -*- coding: utf-8 -*-
"""Login services: the MemHub Core, and the SSH tunnel to a Core on another host.

macOS uses LaunchAgents, Linux systemd user units. Both start at login and
restart on failure. No credentials are copied into a service file: like any
dual-mem process, the Core reads model keys from ``~/.dual_mem/config.yaml``
(a login service does not see variables exported in a shell profile), and the
tunnel authenticates with the user's own SSH keys / ``~/.ssh/config``.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dual_mem.memhub.config import hub_root

LABEL = "com.dual-mem.hub"
SYSTEMD_UNIT = "dual-mem-hub.service"
TUNNEL_LABEL = "com.dual-mem.hub-tunnel"
TUNNEL_SYSTEMD_UNIT = "dual-mem-hub-tunnel.service"

Runner = Callable[[list[str]], dict[str, Any]]


@dataclass(frozen=True)
class _Service:
    label: str
    unit: str
    description: str
    log_name: str
    # The Core exits 0 only when asked to stop; ssh exits whenever the link drops.
    restart_always: bool


CORE = _Service(LABEL, SYSTEMD_UNIT, "dual-mem MemHub Core", "serve.log", False)
TUNNEL = _Service(
    TUNNEL_LABEL, TUNNEL_SYSTEMD_UNIT, "dual-mem MemHub SSH tunnel", "tunnel.log", True
)


def _argv(hub_dir: Path, host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "dual_mem.cli",
        "hub",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
        "--hub-dir",
        str(hub_dir),
    ]


def tunnel_argv(
    target: str, *, local_port: int, remote_port: int, ssh_args: list[str] | None = None
) -> list[str]:
    """``ssh -N`` forwarding local loopback to the remote Core's loopback.

    BatchMode never prompts (a service has no terminal); ExitOnForwardFailure and
    keep-alives make a broken link exit, so the service manager restarts it.
    """
    return [
        "ssh",
        "-N",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        *(ssh_args or []),
        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        target,
    ]


def launchd_plist_path(label: str = LABEL) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def systemd_unit_path(unit: str = SYSTEMD_UNIT) -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "systemd" / "user" / unit


def _plist(service: _Service, argv: list[str], workdir: Path, logs: Path) -> bytes:
    return plistlib.dumps(
        {
            "Label": service.label,
            "ProgramArguments": argv,
            "RunAtLoad": True,
            "KeepAlive": True if service.restart_always else {"SuccessfulExit": False},
            # launchd throttles respawns to this interval (seconds).
            "ThrottleInterval": 10,
            "WorkingDirectory": str(workdir),
            "StandardOutPath": str(logs / service.log_name),
            "StandardErrorPath": str(logs / service.log_name),
        }
    )


def _unit(service: _Service, argv: list[str], workdir: Path, logs: Path) -> str:
    return "\n".join(
        [
            "[Unit]",
            f"Description={service.description}",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={workdir}",
            f"ExecStart={shlex.join(argv)}",
            f"Restart={'always' if service.restart_always else 'on-failure'}",
            f"RestartSec={10 if service.restart_always else 5}",
            f"StandardOutput=append:{logs / service.log_name}",
            f"StandardError=append:{logs / service.log_name}",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def render_launchd_plist(
    *, hub_dir: Path, host: str = "127.0.0.1", port: int = 8765, log_dir: Path | None = None
) -> bytes:
    return _plist(CORE, _argv(hub_dir, host, port), hub_dir, log_dir or hub_dir / "logs")


def render_systemd_unit(
    *, hub_dir: Path, host: str = "127.0.0.1", port: int = 8765, log_dir: Path | None = None
) -> str:
    return _unit(CORE, _argv(hub_dir, host, port), hub_dir, log_dir or hub_dir / "logs")


def _run(argv: list[str]) -> dict[str, Any]:
    result = subprocess.run(argv, capture_output=True, text=True)
    return {"command": argv, "returncode": result.returncode, "stderr": result.stderr.strip()}


def _install(service: _Service, argv: list[str], root: Path, runner: Runner) -> dict[str, Any]:
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    if sys.platform == "darwin":
        path = launchd_plist_path(service.label)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_plist(service, argv, root, logs))
        actions = [
            runner(["launchctl", "unload", str(path)]),
            runner(["launchctl", "load", str(path)]),
        ]
    else:
        path = systemd_unit_path(service.unit)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_unit(service, argv, root, logs), encoding="utf-8")
        actions = [
            runner(["systemctl", "--user", "daemon-reload"]),
            # restart: a re-run of connect must pick up a changed port or target.
            runner(["systemctl", "--user", "enable", service.unit]),
            runner(["systemctl", "--user", "restart", service.unit]),
        ]
    return {
        "service_file": str(path),
        "log": str(logs / service.log_name),
        "actions": actions,
    }


def _uninstall(service: _Service, runner: Runner) -> dict[str, Any]:
    if sys.platform == "darwin":
        path = launchd_plist_path(service.label)
        actions = [runner(["launchctl", "unload", str(path)])] if path.is_file() else []
    else:
        path = systemd_unit_path(service.unit)
        actions = (
            [runner(["systemctl", "--user", "disable", "--now", service.unit])]
            if path.is_file()
            else []
        )
    removed = path.is_file()
    if removed:
        path.unlink()
    return {"service_file": str(path), "removed": removed, "actions": actions}


def install_service(
    *,
    hub_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    runner: Runner = _run,
) -> dict[str, Any]:
    """Write and load the Core login service for this Hub directory."""
    root = hub_root(hub_dir)
    return _install(CORE, _argv(root, host, port), root, runner)


def uninstall_service(*, runner: Runner = _run) -> dict[str, Any]:
    return _uninstall(CORE, runner)


def install_tunnel(
    target: str,
    *,
    local_port: int,
    remote_port: int,
    ssh_args: list[str] | None = None,
    hub_dir: str | Path | None = None,
    runner: Runner = _run,
) -> dict[str, Any]:
    """Keep ``ssh -N -L`` to a remote Core running across logins and dropped links."""
    argv = tunnel_argv(
        target, local_port=local_port, remote_port=remote_port, ssh_args=ssh_args
    )
    return _install(TUNNEL, argv, hub_root(hub_dir), runner)


def uninstall_tunnel(*, runner: Runner = _run) -> dict[str, Any]:
    return _uninstall(TUNNEL, runner)
