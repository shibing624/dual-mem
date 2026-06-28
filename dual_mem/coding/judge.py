# -*- coding: utf-8 -*-
"""Coding memory scene judge — LLM binary classifier (coding vs chat).

Fail-safe: any LLM error or parse failure → False (fallback to chat path).
"""
import json
import logging
from typing import Any, Dict, List, Optional

from dual_mem.providers.llm import LLMClient
from dual_mem.coding.preproc import extract_tool_summary

logger = logging.getLogger("dual_mem.coding.judge")

_CLASSIFY_PROMPT = """\
You are a single-label scene classifier. Decide the DOMINANT scene of the entire
conversation chunk passed to you.

Output exactly one of:
  "coding"  — user is primarily doing engineering work (code/files/commands/
              deploy/debug). Tool calls are doing real work, and instructions/
              conventions/decisions/learnings may emerge.
  "chat"    — user is primarily having casual conversation, sharing personal
              info, asking factual Q&A unrelated to dev/ops. Tool calls (if any)
              are incidental.

Rules:
- Look at the WHOLE chunk, not individual turns.
- A few stray tool calls inside a clearly chat chunk should still be "chat".
- When in doubt, prefer "chat".

Conversation summary (turns with user query + tool names used):
{turns_block}

Output strict JSON only, no markdown:
{{"is_coding": true/false, "reason": "..."}}
"""


async def classify_messages_is_coding(
    messages: List[dict],
    llm: LLMClient,
    *,
    max_tokens: int = 200,
    temperature: float = 0.1,
) -> bool:
    """Classify if messages are coding or chat. Fail-safe → False."""
    if not messages:
        return False

    summary = extract_tool_summary(messages)
    turns_block = _format_turns(summary)
    prompt = _CLASSIFY_PROMPT.format(turns_block=turns_block)

    try:
        resp = await llm.chat_json(system=prompt, user="", temperature=temperature)
        if isinstance(resp, dict) and "is_coding" in resp:
            is_coding = bool(resp["is_coding"])
            logger.info("[coding-judge] is_coding=%s reason=%r", is_coding, resp.get("reason", ""))
            return is_coding
    except Exception as e:
        logger.warning("[coding-judge] classify failed: %s; defaulting to chat", e)

    return False


def _format_turns(summary: List[Dict[str, Any]]) -> str:
    if not summary:
        return "(empty)"
    rows = []
    for t in summary:
        user = (t.get("user") or "")[:200]
        tools = t.get("tools") or []
        rows.append(
            f"- turn {t.get('turn', '?')}: user={user!r}, tools={tools}"
            + (", has_tool_result=True" if t.get("has_tool_result") else "")
        )
    return "\n".join(rows)
