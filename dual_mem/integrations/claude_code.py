# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Claude Code Hook 适配器（基于 dual_mem）。

Claude Code Hook 是一次性命令行调用：stdin 收 JSON payload，stdout 回 JSON 结果。
子命令：
  dual-mem-hook search   UserPromptSubmit hook — 搜记忆注入 additionalContext
  dual-mem-hook ingest   Stop hook — 提取对话写入记忆

环境变量：DUAL_MEM_USER_ID（必填）/ DUAL_MEM_AGENT_ID（默认 claude-code）/
DUAL_MEM_STORAGE_DIR / DUAL_MEM_MODE。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

logger = logging.getLogger("dual_mem.integrations.claude_code")


def _make_backend() -> Any:
    from dual_mem.integrations._base import AsyncRunner, MemoryBackend
    from dual_mem.config import Settings

    user_id = os.environ.get("DUAL_MEM_USER_ID", "").strip()
    if not user_id:
        raise RuntimeError("DUAL_MEM_USER_ID is required for the Claude Code hook")
    mode = os.environ.get("DUAL_MEM_MODE", "").strip().lower()
    mode = "dual" if mode in ("dual", "ultra", "pro") else "system1"
    storage_dir = os.environ.get("DUAL_MEM_STORAGE_DIR")
    settings = Settings(storage_dir=storage_dir, mode=mode) if storage_dir else Settings(mode=mode)
    backend = MemoryBackend(settings=settings)
    runner = AsyncRunner(backend.client)
    return backend, runner, user_id, os.environ.get("DUAL_MEM_AGENT_ID", "claude-code").strip() or "claude-code"


def handle_search(payload: dict, *, limit: int = 10, min_score: float = 0.4) -> dict:
    """UserPromptSubmit hook：返回 {"additionalContext": "<relevant-memories>…"}。"""
    backend, runner, user_id, agent_id = _make_backend()
    try:
        prompt = (payload.get("user_prompt") or "").strip()
        if not prompt or len(prompt) < 3:
            return {"additionalContext": ""}
        result = runner.run(
            backend.search(
                query=prompt, user_id=user_id,
                agent_ids=[agent_id], limit=limit, min_score=min_score, intention_limit=0,
            )
        )
        block = _render(result, user_id, agent_id)
        return {"additionalContext": block}
    except Exception as exc:  # hook 失败不能阻断会话
        logger.warning("claude_code search hook failed: %s", exc)
        return {"additionalContext": ""}
    finally:
        runner.close()


def _render(result: Any, user_id: str, agent_id: str) -> str:
    from dual_mem.integrations._base import format_memories_for_prompt
    items = result.memories.flatten(limit=10)
    return format_memories_for_prompt(items, max_chars=2000)


def handle_ingest(payload: dict) -> dict:
    """Stop hook：把 transcript 里的 user/assistant 轮次写入记忆。"""
    backend, runner, user_id, agent_id = _make_backend()
    try:
        transcript = payload.get("transcript") or []
        messages: list[dict] = []
        for turn in transcript:
            role = (turn.get("role") or "").lower()
            text = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and text:
                messages.append({"role": role, "content": text})
        if len(messages) < 2:
            return {"status": "ok", "memories_added": 0}
        runner.run(
            backend.add(messages=messages, user_id=user_id, agent_id=agent_id)
        )
        return {"status": "ok", "memories_added": 1}
    except Exception as exc:
        logger.warning("claude_code ingest hook failed: %s", exc)
        return {"status": "error", "memories_added": 0, "error": str(exc)}
    finally:
        runner.close()


def _read_stdin() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def _write_stdout(data: dict) -> None:
    sys.stdout.write(json.dumps(data, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(prog="dual-mem-hook", description="dual-mem Claude Code hooks")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("search", help="UserPromptSubmit hook")
    sp.add_argument("--limit", type=int, default=10)
    sp.add_argument("--min-score", type=float, default=0.4)

    sub.add_parser("ingest", help="Stop hook")

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    payload = _read_stdin()
    if args.command == "search":
        _write_stdout(handle_search(payload, limit=args.limit, min_score=args.min_score))
    elif args.command == "ingest":
        _write_stdout(handle_ingest(payload))


if __name__ == "__main__":
    main()
