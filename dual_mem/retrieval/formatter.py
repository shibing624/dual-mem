# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Formats three-route search results into an LLM context string (profile ->
proactive -> normal), expanding evolution chains and truncating long raw memories.
"""
_GROUP_TITLES = [
    ("profile", "【画像 Profile】"),
    ("proactive", "【主动 Proactive】"),
    ("normal", "【常规记忆 Normal】"),
]


def _format_item(item: dict, raw_truncate: int) -> str:
    """Format one memory item as a context line, expanding any evolution chain."""
    content = item["content"]
    if item["category"] == "raw" and len(content) > raw_truncate:
        content = content[:raw_truncate] + "…"

    lines = [f"- [{item['category']}] {content}"]
    chain = item.get("evolution_chain")
    if chain:
        lines.append("  演化历史（最新→最旧）:")
        for idx, ver in enumerate(chain, start=1):
            lines.append(f"    {idx}. {ver['content']}")
    return "\n".join(lines)


def format_memories(result: dict, raw_truncate: int = 800) -> str:
    """Join non-empty profile/proactive/normal groups into a single context string."""
    blocks: list[str] = []
    for key, title in _GROUP_TITLES:
        items = result.get(key) or []
        if not items:
            continue
        body = "\n".join(_format_item(it, raw_truncate) for it in items)
        blocks.append(f"{title}\n{body}")
    return "\n\n".join(blocks)
