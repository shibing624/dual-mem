# -*- coding: utf-8 -*-
"""Tool-trajectory preprocessing — structural message routing, zero LLM.

Messages are OpenAI-style dicts: role/content, optional assistant ``tool_calls``
(flat ``name``/``arguments`` or nested under ``function``), and ``role="tool"``
results.
"""
import json
from typing import Any

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


def _text(content: Any) -> str:
    """Text of a message content: strings as-is, structured parts as JSON."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def clip_text(text: str, max_chars: int) -> str:
    """Keep head and tail of an oversized tool result; the middle is elided."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n[… {len(text) - 2 * half} chars elided …]\n{text[-half:]}"


def clip_tool_results(messages: list[dict], max_chars: int) -> list[dict]:
    """Bound each ``role="tool"`` result for model input.

    Long build logs and file dumps dominate a trajectory; the decisions and the
    error lines that matter sit at the head (command echo) and tail (exit status,
    traceback). The lossless trajectory stays in L0_RAW.
    """
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "tool":
            m = dict(m, content=clip_text(_text(m.get("content")), max_chars))
        out.append(m)
    return out


def _call_text(call: dict) -> str:
    function = call.get("function") or {}
    name = call.get("name") or function.get("name") or ""
    args = call.get("arguments", function.get("arguments"))
    return f"{name}({_text(args)})"


def render_transcript(messages: list[dict]) -> str:
    """Role-labelled text of a full trajectory (tool calls and results included).

    Uses the same ``[role]: text`` lines as chat dialogue, so the raw route and
    excerpting work unchanged on coding sources.
    """
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role") or "user")
        text = _text(m.get("content")).strip()
        if text:
            lines.append(f"[{role}]: {text}")
        for call in m.get("tool_calls") or []:
            lines.append(f"[{role}]: {_call_text(call)}")
    return "\n".join(lines)
