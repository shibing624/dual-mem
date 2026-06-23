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


def _hit_text(hit: dict) -> str:
    return str(hit.get("content") or hit.get("memory") or "")


def _hit_timestamp(hit: dict) -> int | None:
    for key in ("memory_at", "gmt_created"):
        val = hit.get(key)
        if isinstance(val, int) and val > 0:
            return val
    created_at = hit.get("created_at")
    if isinstance(created_at, str) and created_at:
        try:
            from datetime import datetime

            if created_at.endswith("Z"):
                created_at = created_at[:-1] + "+00:00"
            return int(datetime.fromisoformat(created_at).timestamp())
        except ValueError:
            return None
    return None


def format_search_hit_line(
    hit: dict,
    *,
    prefix: str = "- ",
    include_date: bool = True,
    include_evolution: bool = False,
) -> str:
    """Format one flat search hit (``memory`` or ``content`` key) for QA prompts."""
    category = hit.get("category", "")
    text = _hit_text(hit)
    cat_part = f"[{category}] " if category else ""
    date_suffix = ""
    if include_date:
        mem_date = format_memory_timestamp(_hit_timestamp(hit))
        if mem_date:
            date_suffix = f" (conversation date: {mem_date})"
    line = f"{prefix}{cat_part}{text}{date_suffix}"

    chain = hit.get("evolution_chain")
    if not include_evolution or not chain or len(chain) <= 1:
        return line

    older = list(reversed(chain[1:]))
    tail = "\n".join(
        f"{prefix}  · {format_memory_timestamp(_hit_timestamp(ver)) or '?'}: {ver.get('content', '')}"
        for ver in older
    )
    current = f"{prefix}  → current: {chain[0].get('content', text)}"
    return f"{line}\n{tail}\n{current}"


def collect_evolution_chains(hits: list[dict]) -> list[dict]:
    """Return unique hits that carry a multi-step ``evolution_chain`` (newest-first)."""
    seen_heads: set[str] = set()
    chains: list[dict] = []
    for hit in hits:
        chain = hit.get("evolution_chain")
        if not chain or len(chain) <= 1:
            continue
        head_id = str(chain[0].get("node_id") or hit.get("memory_id") or hit.get("id") or "")
        if head_id and head_id in seen_heads:
            continue
        if head_id:
            seen_heads.add(head_id)
        chains.append(hit)
    chains.sort(key=lambda h: float(h.get("score") or 0), reverse=True)
    return chains


def format_evolution_timeline(
    hits: list[dict],
    *,
    header: str = "Preference evolution (oldest → newest):",
    max_chains: int = 5,
    max_versions: int = 4,
) -> str:
    """Render evolution chains as a compact oldest→newest timeline (generic, no benchmark ids)."""
    chains = collect_evolution_chains(hits)[:max_chains]
    if not chains:
        return ""

    blocks: list[str] = [header]
    for hit in chains:
        chain = hit["evolution_chain"][:max_versions]
        ordered = list(reversed(chain))
        topic = _hit_text(hit)[:80].strip() or "topic"
        lines = [f"  · {topic}"]
        for idx, ver in enumerate(ordered):
            date = format_memory_timestamp(
                _hit_timestamp(ver) or ver.get("memory_at") or ver.get("gmt_created")
            )
            label = "CURRENT" if idx == len(ordered) - 1 else (date or "?")
            prefix = "  →" if idx == len(ordered) - 1 else "    "
            lines.append(f"{prefix} {label}: {ver.get('content', '')}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def chain_node_ids(hit: dict) -> set[str]:
    """All node ids appearing in a hit's evolution chain."""
    chain = hit.get("evolution_chain") or []
    ids: set[str] = set()
    for ver in chain:
        nid = ver.get("node_id")
        if nid:
            ids.add(str(nid))
    mid = hit.get("memory_id") or hit.get("id")
    if mid:
        ids.add(str(mid))
    return ids

