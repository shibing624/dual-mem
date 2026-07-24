# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: OpenClaw MemoryProvider 插件（与 Hermes 同源契约，基于 dual_mem）。

OpenClaw 与 Hermes 共用同一套 memory provider 接口（prefetch / sync_turn /
on_session_end / shutdown），因此本实现直接复用 _SyncMemoryProvider 的 machinery，
仅替换配置来源与默认 agent_id。仍基于 dual_mem 内嵌 MemoryClient。
"""
from __future__ import annotations

import logging
import os
from typing import Any

from dual_mem.integrations._base import _SyncMemoryProvider

logger = logging.getLogger("dual_mem.integrations.openclaw")

try:  # pragma: no cover - depends on host runtime
    from agent.memory_provider import MemoryProvider as _OpenClawMemoryProvider
except Exception:  # noqa: BLE001 - optional host dependency
    class _OpenClawMemoryProvider:  # type: ignore[no-redef]
        pass


class DualMemOpenClawProvider(_SyncMemoryProvider, _OpenClawMemoryProvider):
    """OpenClaw 原生 memory 插件：被动注入 + 异步写入，基于 dual_mem。"""

    name = "dual-mem-openclaw"

    def is_available(self) -> bool:
        return bool(os.environ.get("DUAL_MEM_USER_ID", os.environ.get("OPENCLAW_USER_ID")))

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        with self._lock:
            if self._initialized:
                return
            self._user_id = (
                os.environ.get("DUAL_MEM_USER_ID", os.environ.get("OPENCLAW_USER_ID", "")).strip()
            )
            self._agent_id = (
                os.environ.get("DUAL_MEM_AGENT_ID", os.environ.get("OPENCLAW_AGENT_ID", "openclaw")).strip()
                or "openclaw"
            )
            self._mode = _resolve_mode(
                os.environ.get("DUAL_MEM_MODE", os.environ.get("OPENCLAW_MODE", ""))
            )
            if not self._user_id:
                logger.error("[openclaw] DUAL_MEM_USER_ID not set; provider disabled")
                self._initialized = True
                return
            super().initialize(session_id, **kwargs)

    def get_config_schema(self) -> list[dict]:
        return [
            {
                "key": "user_id",
                "label": "User ID",
                "description": "Your unique memory namespace identifier",
                "env_var": "DUAL_MEM_USER_ID",
                "required": True,
                "secret": False,
            },
            {
                "key": "agent_id",
                "label": "Agent ID",
                "description": "Agent identifier for memory isolation",
                "env_var": "DUAL_MEM_AGENT_ID",
                "required": False,
                "secret": False,
                "default": "openclaw",
            },
            {
                "key": "mode",
                "label": "Processing Mode",
                "description": "system1 (fast) / dual (system1 + System2 graph)",
                "env_var": "DUAL_MEM_MODE",
                "required": False,
                "secret": False,
                "default": "system1",
                "choices": ["system1", "dual"],
            },
        ]

    def save_config(self, values: dict, home: str) -> None:
        env_file = os.path.join(home, ".env")
        env_lines: list[str] = []
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                env_lines = f.readlines()
        env_map = {
            "user_id": "DUAL_MEM_USER_ID",
            "agent_id": "DUAL_MEM_AGENT_ID",
            "mode": "DUAL_MEM_MODE",
        }
        for key, env_var in env_map.items():
            val = values.get(key, "")
            if val:
                env_lines = [
                    line for line in env_lines
                    if not line.strip().startswith(f"{env_var}=")
                ]
                env_lines.append(f"{env_var}={val}\n")
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(env_lines)

    def register(self, ctx: Any) -> None:
        ctx.register_memory_provider(self)


def _resolve_mode(raw: str) -> str:
    raw = (raw or "").strip().lower()
    if raw in ("dual", "ultra", "pro"):
        return "dual"
    return "system1"
