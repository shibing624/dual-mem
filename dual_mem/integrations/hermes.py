# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Hermes Agent MemoryProvider 插件（基于 dual_mem 内嵌 MemoryClient）。

生命周期对齐 Hermes 官方契约：
  is_available() → initialize(session_id) → [prefetch / sync_turn 反复]
  → [on_pre_compress / on_session_end] → shutdown()

与 tmp/plugins/native/hermes 的区别：本实现直接驱动 dual_mem 的 MemoryClient
（不再经过 one-memory HTTP server），用 DUAL_MEM_* 环境变量配置。
"""
from __future__ import annotations

import logging
import os
from typing import Any

from dual_mem.integrations._base import _SyncMemoryProvider

logger = logging.getLogger("dual_mem.integrations.hermes")

# 继承 Hermes 的 MemoryProvider ABC（安装 hermes 时 isinstance 检查能过）；
# 未安装则回落到占位 base，逻辑一致。
try:  # pragma: no cover - depends on host runtime
    from agent.memory_provider import MemoryProvider as _HermesMemoryProvider
except Exception:  # noqa: BLE001 - optional host dependency
    class _HermesMemoryProvider:  # type: ignore[no-redef]
        pass


def _resolve_mode(raw: str) -> str:
    """把 Hermes/OpenClaw 的 lite/pro/ultra 映射到 dual_mem 的 system1/dual。"""
    raw = (raw or "").strip().lower()
    if raw in ("dual", "ultra", "pro"):
        return "dual"
    return "system1"


class DualMemHermesProvider(_SyncMemoryProvider, _HermesMemoryProvider):
    """Hermes 原生 memory 插件：被动注入 + 异步写入，基于 dual_mem。"""

    name = "dual-mem"

    def is_available(self) -> bool:
        return bool(os.environ.get("DUAL_MEM_USER_ID", os.environ.get("HY_MEMORY_USER_ID")))

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        with self._lock:
            if self._initialized:
                return
            self._user_id = (
                os.environ.get("DUAL_MEM_USER_ID", os.environ.get("HY_MEMORY_USER_ID", "")).strip()
            )
            self._agent_id = (
                os.environ.get("DUAL_MEM_AGENT_ID", os.environ.get("HY_MEMORY_AGENT_ID", "hermes")).strip()
                or "hermes"
            )
            self._mode = _resolve_mode(
                os.environ.get("DUAL_MEM_MODE", os.environ.get("HY_MEMORY_MODE", ""))
            )
            if not self._user_id:
                logger.error("[hermes] DUAL_MEM_USER_ID not set; provider disabled")
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
                "default": "hermes",
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

    def save_config(self, values: dict, hermes_home: str) -> None:
        env_file = os.path.join(hermes_home, ".env")
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
        logger.info("[hermes] Config saved to %s", env_file)

    def register(self, ctx: Any) -> None:
        """Hermes 插件注册入口：把本 Provider 注册进 Hermes runtime。"""
        ctx.register_memory_provider(self)
