# -*- coding: utf-8 -*-
"""Coding memory preprocessing — rule-based, zero LLM.

Adapted for dual_mem's dict message format: messages are list[dict] with
keys role/content, optional tool_calls (list[dict] with name/arguments),
optional tool_call_id / tool_name for role=tool messages.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

PATH_KEYS: Tuple[str, ...] = (
    "path", "file_path", "filename", "file", "filepath",
    "target_path", "notebook_path", "src_path", "dst_path",
)

DEFAULT_MAX_TOOL_RESULT_BYTES = 2048
DEFAULT_HEAD_BYTES = 1024
DEFAULT_TAIL_BYTES = 512


def has_any_tool_message(messages: List[dict]) -> bool:
    """Check if messages contain tool messages or assistant tool_calls."""
    for m in messages:
        role = m.get("role", "")
        if role == "tool":
            return True
        if role == "assistant" and m.get("tool_calls"):
            return True
    return False


def strip_tool_messages(messages: List[dict]) -> List[dict]:
    """Remove tool messages + tool_calls field for chat-path consumption."""
    out: List[dict] = []
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


def truncate_tool_result_text(
    text: str,
    max_bytes: int = DEFAULT_MAX_TOOL_RESULT_BYTES,
    head_bytes: int = DEFAULT_HEAD_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
) -> str:
    """Head-tail truncation for long tool output."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if len(text) <= max_bytes:
        return text
    if head_bytes + tail_bytes >= len(text):
        return text
    omitted = len(text) - head_bytes - tail_bytes
    return text[:head_bytes] + f"\n\n[...{omitted} bytes omitted...]\n\n" + text[-tail_bytes:]


def truncate_messages(
    messages: List[dict],
    max_bytes: int = DEFAULT_MAX_TOOL_RESULT_BYTES,
) -> List[dict]:
    """Truncate all tool message contents."""
    out = []
    for m in messages:
        if m.get("role") == "tool" and len(m.get("content", "")) > max_bytes:
            out.append({**m, "content": truncate_tool_result_text(m["content"], max_bytes)})
        else:
            out.append(m)
    return out


def extract_files(messages: List[dict]) -> List[str]:
    """Extract file paths from tool_calls.arguments."""
    paths: set = set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                continue
            for k in PATH_KEYS:
                v = args.get(k)
                if isinstance(v, str) and v:
                    paths.add(v)
    return sorted(paths)


def extract_tool_summary(messages: List[dict]) -> List[Dict[str, Any]]:
    """Turn-level simplified view for LLM judge/extractor."""
    out: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    def _flush():
        if cur is not None:
            seen = set()
            cur["tools"] = [t for t in cur["tools"] if not (t in seen or seen.add(t))]
            out.append(cur)

    for m in messages:
        role = m.get("role", "")
        if role == "user" and role != "tool":
            _flush()
            cur = {
                "turn": len(out),
                "user": (m.get("content") or "").strip(),
                "tools": [],
                "has_tool_result": False,
            }
        else:
            if cur is None:
                cur = {"turn": len(out), "user": "", "tools": [], "has_tool_result": False}
            if role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    name = tc.get("name") or tc.get("function", {}).get("name")
                    if name:
                        cur["tools"].append(name)
            if role == "tool":
                cur["has_tool_result"] = True
                tn = m.get("tool_name") or m.get("name")
                if tn:
                    cur["tools"].append(tn)

    _flush()
    return out
