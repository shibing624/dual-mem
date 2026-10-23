# -*- coding: utf-8 -*-
"""Coding memory preprocessing — structural message routing, zero LLM.

Messages are OpenAI-style dicts: role/content, optional assistant ``tool_calls``
(flat ``name``/``arguments`` or nested under ``function``), and ``role="tool"``
results.
"""
import json

PATH_KEYS: tuple[str, ...] = (
    "path",
    "file_path",
    "filename",
    "file",
    "filepath",
    "target_path",
    "notebook_path",
    "src_path",
    "dst_path",
)


def has_any_tool_message(messages: list[dict]) -> bool:
    """Check if messages contain tool messages or assistant tool_calls."""
    return any(
        m.get("role") == "tool" or (m.get("role") == "assistant" and m.get("tool_calls"))
        for m in messages
    )


def strip_tool_messages(messages: list[dict]) -> list[dict]:
    """Remove tool messages + tool_calls field for chat-path consumption."""
    out: list[dict] = []
    for m in messages:
        role = m.get("role", "")
        if role == "tool":
            continue
        if role == "assistant" and m.get("tool_calls"):
            content = (m.get("content") or "").strip()
            if content:
                out.append({"role": "assistant", "content": content})
            continue
        out.append(m)
    return out


def extract_files(messages: list[dict]) -> list[str]:
    """File paths named by the well-known path arguments of tool calls."""
    paths: set[str] = set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            args = tc.get("arguments", tc.get("function", {}).get("arguments"))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if not isinstance(args, dict):
                continue
            for k in PATH_KEYS:
                v = args.get(k)
                if isinstance(v, str) and v:
                    paths.add(v)
    return sorted(paths)
