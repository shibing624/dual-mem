"""Generate complete coding candidates; JEV owns all semantic decisions."""

import json
import math
from typing import Any

from dual_mem.coding.preproc import extract_files
from dual_mem.coding.types import (
    BOUNDARY_SCOPES,
    CodingDecisionError,
    CodingMemoryDraft,
)
from dual_mem.providers.llm import LLMClient

_EXTRACT_PROMPT = """Generate candidate engineering memories from the source JSON.
Only produce candidate text, never decide persistence value, knownness, updates,
actions or targets. Include every potentially useful explicit engineering fact,
constraint, decision, changed solution and distinct event. Do not filter trivial
or already documented information: JEV will decide its value and coverage.
Preserve complete commands, paths, numbers, reasons, tool arguments/results,
environment boundaries and qualifications. Do not invent missing details.
Write every text field in the language of the source conversation.

Output a JSON object {"candidates": [...]}, each candidate an object with:
- task: string, a self-contained retrieval question
- search_keys: array of strings, alternative phrasings of the task
- solution: string, the complete content
- boundary_envs: string, environment constraints in one sentence ("" if none)
- boundary_scope: one of "strict" | "project" | "user" | "global"
- confidence: number in [0, 1]
Return an empty candidates array only when the source has no engineering content.
"""


class CodingMemoryExtractor:
    def __init__(self, *, llm: LLMClient) -> None:
        self.llm = llm

    async def extract(
        self,
        messages: list[dict],
        *,
        user_id: str,
        agent_id: str = "",
        app_id: str = "default",
        workspace_id: str | None = None,
        branch: str | None = None,
        session_id: str | None = None,
    ) -> list[CodingMemoryDraft]:
        # An object envelope, not a bare array: JSON mode (llm_json_mode, on by
        # default) only admits objects, and the parser drops anything else.
        data = await self.llm.chat_json(
            system=_EXTRACT_PROMPT,
            user=json.dumps(messages, ensure_ascii=False),
            temperature=0.1,
        )
        items = data.get("candidates") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise CodingDecisionError(
                'coding extraction requires a {"candidates": [...]} object'
            )
        files = extract_files(messages)
        drafts: list[CodingMemoryDraft] = []
        for item in items:
            if not isinstance(item, dict):
                raise CodingDecisionError("invalid coding candidate object")
            task = self._text(item, "task", required=True)
            solution = self._text(item, "solution", required=True)
            keys = item.get("search_keys", [])
            if not isinstance(keys, list) or any(
                not isinstance(key, str) or not key.strip() for key in keys
            ):
                raise CodingDecisionError("invalid coding candidate search_keys")
            scope = item.get("boundary_scope", "project")
            confidence = item.get("confidence", 0.7)
            if scope not in BOUNDARY_SCOPES:
                raise CodingDecisionError("invalid coding candidate boundary_scope")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise CodingDecisionError("invalid coding candidate confidence")
            drafts.append(
                CodingMemoryDraft(
                    task=task,
                    search_keys=list(keys),
                    solution=solution,
                    boundary_envs=self._text(item, "boundary_envs"),
                    boundary_scope=scope,
                    confidence=float(confidence),
                    user_id=user_id,
                    agent_id=agent_id,
                    app_id=app_id,
                    workspace_id=workspace_id,
                    branch=branch,
                    session_id=session_id,
                    files=list(files),
                )
            )
        return drafts

    @staticmethod
    def _text(item: dict[str, Any], name: str, *, required: bool = False) -> str:
        text = item.get(name, "")
        if not isinstance(text, str) or (required and not text.strip()):
            raise CodingDecisionError(f"invalid coding candidate {name}")
        return text
