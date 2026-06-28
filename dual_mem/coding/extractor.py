# -*- coding: utf-8 -*-
"""Coding memory extractor — LLM extracts durable engineering memories.

Value bar is strict: trivial ops (rename, git status) should NOT become memories.
Only extract problems likely to recur with non-trivial solutions.
"""
import json
import logging
from typing import List, Optional

from dual_mem.providers.llm import LLMClient
from dual_mem.coding.preproc import truncate_messages, extract_files
from dual_mem.coding.types import CodingMemoryDraft, BOUNDARY_SCOPES

logger = logging.getLogger("dual_mem.coding.extractor")

_EXTRACT_PROMPT = """\
You extract durable engineering memories from a coding session segment.
Each memory is a (task, search_keys, solution, boundary_envs, boundary_scope) bundle.

VALUE BAR — DO NOT cross this lightly.
A memory MUST satisfy ALL of:
  (a) Solves a problem likely to recur for the same user
  (b) Solution is non-trivial: combines multiple pieces of info, OR carries
      a decision rationale, OR captures a hard-won workaround
  (c) A future engineer would prefer looking it up over re-discovering it

DO NOT extract:
  - Trivial single-step ops ("rename file", "git status", "run ls")
  - One-off bugs already fixed with no transferable lesson
  - Information already in the repo

MEMORY FIELDS:
  task          — How a user would later ASK. Self-contained.
  search_keys   — 0-5 alternative phrasings for retrieval.
  solution      — Complete content: commands, paths, steps, gotchas.
  boundary_envs — Runtime/SDK/library versions, config paths. Empty if agnostic.
  boundary_scope — strict|project|user|global (narrower preferred).
  confidence    — 0.0-1.0 (1.0=explicit rule, 0.7=inferred, 0.5=uncertain).

Conversation:
{conversation}

Output JSON array of memory objects. Empty array if nothing worth extracting:
[{{"task": "...", "search_keys": ["..."], "solution": "...", "boundary_envs": "", "boundary_scope": "project", "confidence": 0.7}}]
"""


class CodingMemoryExtractor:
    """Extract coding memories from tool-use conversations."""

    def __init__(self, *, llm: LLMClient, tool_result_max_bytes: int = 2048):
        self.llm = llm
        self.tool_result_max_bytes = tool_result_max_bytes

    async def extract(
        self,
        messages: List[dict],
        *,
        user_id: str,
        agent_id: str = "default_agent",
        workspace_id: Optional[str] = None,
        branch: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[CodingMemoryDraft]:
        """Extract coding memory drafts from messages."""
        truncated = truncate_messages(messages, max_bytes=self.tool_result_max_bytes)
        files = extract_files(messages)
        conversation = self._format_conversation(truncated)
        prompt = _EXTRACT_PROMPT.format(conversation=conversation)

        try:
            data = await self.llm.chat_json(system=prompt, user="", temperature=0.1)
        except Exception as e:
            logger.warning("[coding-extract] LLM failed: %s", e)
            return []

        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) else []

        drafts: List[CodingMemoryDraft] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            task = (item.get("task") or "").strip()
            if not task:
                continue
            scope = item.get("boundary_scope", "project")
            if scope not in BOUNDARY_SCOPES:
                scope = "project"
            drafts.append(CodingMemoryDraft(
                task=task,
                search_keys=[str(k) for k in (item.get("search_keys") or []) if k][:5],
                solution=item.get("solution") or "",
                boundary_envs=item.get("boundary_envs") or "",
                boundary_scope=scope,
                confidence=float(item.get("confidence", 0.7)),
                user_id=user_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
                branch=branch,
                session_id=session_id,
                files=files,
            ))

        logger.info("[coding-extract] extracted %d drafts", len(drafts))
        return drafts

    @staticmethod
    def _format_conversation(messages: List[dict]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "?")
            content = (m.get("content") or "")[:500]
            lines.append(f"[{role}] {content}")
            tcs = m.get("tool_calls") or []
            for tc in tcs:
                name = tc.get("name") or tc.get("function", {}).get("name", "?")
                lines.append(f"  → tool: {name}")
        return "\n".join(lines)
