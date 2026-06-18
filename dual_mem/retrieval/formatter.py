"""把三路检索结果拼成 LLM 上下文字符串。

顺序 profile → proactive → normal，演化链按 latest→oldest 展开多版本，
raw 类记忆内容超长截断。空分组跳过。
"""

_GROUP_TITLES = [
    ("profile", "【画像 Profile】"),
    ("proactive", "【主动 Proactive】"),
    ("normal", "【常规记忆 Normal】"),
]


def _format_item(item: dict, raw_truncate: int) -> str:
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
    blocks: list[str] = []
    for key, title in _GROUP_TITLES:
        items = result.get(key) or []
        if not items:
            continue
        body = "\n".join(_format_item(it, raw_truncate) for it in items)
        blocks.append(f"{title}\n{body}")
    return "\n\n".join(blocks)
