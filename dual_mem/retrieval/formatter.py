# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Formats three-route search results into an LLM context string (profile ->
proactive -> normal). Evolution chains expose only the current (is_latest head) fact by
default; superseded versions are optional. Memory timestamps are labeled as conversation
dates so QA models do not confuse them with "today".
"""
from __future__ import annotations

from datetime import datetime, timezone

_GROUP_TITLES = [
    ("profile", "【画像 Profile】"),
    ("proactive", "【主动 Proactive】"),
    ("normal", "【常规记忆 Normal】"),
]

_DATE_PREAMBLE = (
    "Note: dates on memories are when the user said each fact in past conversations, "
    "not today's date.\n\n"
)


def format_memory_timestamp(ts: int | None) -> str:
    """Format a unix timestamp as YYYY-MM-DD (UTC) for prompt context."""
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _format_item(
    item: dict,
    raw_truncate: int,
    *,
    include_superseded: bool = False,
) -> str:
    """Format one memory item; evolution chains default to current head only."""
    content = item["content"]
    if item["category"] == "raw" and len(content) > raw_truncate:
        content = content[:raw_truncate] + "…"

    mem_date = format_memory_timestamp(item.get("memory_at") or item.get("gmt_created"))
    date_suffix = f" (conversation date: {mem_date})" if mem_date else ""
    lines = [f"- [{item['category']}] {content}{date_suffix}"]

    chain = item.get("evolution_chain")
    if chain and include_superseded and len(chain) > 1:
        lines.append("  (superseded versions — not current):")
        for ver in chain[1:]:
            ver_date = format_memory_timestamp(ver.get("memory_at") or ver.get("gmt_created"))
            date_part = f", {ver_date}" if ver_date else ""
            lines.append(f"    ·{date_part}: {ver['content']}")
    return "\n".join(lines)


def format_memories(
    result: dict,
    raw_truncate: int = 800,
    *,
    include_superseded: bool = False,
    date_preamble: bool = True,
) -> str:
    """Join non-empty profile/proactive/normal groups into a single context string."""
    blocks: list[str] = []
    has_dates = False
    for key, title in _GROUP_TITLES:
        items = result.get(key) or []
        if not items:
            continue
        body_lines: list[str] = []
        for it in items:
            if it.get("memory_at") or it.get("gmt_created"):
                has_dates = True
            body_lines.append(
                _format_item(it, raw_truncate, include_superseded=include_superseded)
            )
        blocks.append(f"{title}\n" + "\n".join(body_lines))

    if not blocks:
        return ""

    text = "\n\n".join(blocks)
    if date_preamble and has_dates:
        return _DATE_PREAMBLE + text
    return text
