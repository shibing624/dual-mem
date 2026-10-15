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

Output a JSON array, each object with:
task (self-contained retrieval question), search_keys (array of alternative
phrasings), solution (complete content), boundary_envs (environment constraints),
boundary_scope (strict|project|user|global), confidence (number in [0,1]).
An empty array is permitted but will be checked by JEV for missed information.
"""


class CodingMemoryExtractor:
    def __init__(self, *, llm: LLMClient, tool_result_max_bytes: int = 2048) -> None:
        self.llm = llm
        # Constructor compatibility only. Semantic evidence is never truncated.
        self.tool_result_max_bytes = tool_result_max_bytes

    async def extract(
        self,
        messages: list[dict],
        *,
        user_id: str,
        agent_id: str = "default_agent",
        app_id: str = "default",
        workspace_id: str | None = None,
        branch: str | None = None,
        session_id: str | None = None,
    ) -> list[CodingMemoryDraft]:
        data = await self.llm.chat_json(
            system=_EXTRACT_PROMPT,
            user=self._format_conversation(messages),
            temperature=0.1,
        )
        if not isinstance(data, list):
            raise CodingDecisionError("coding extraction requires a JSON array")
        files = extract_files(messages)
        drafts: list[CodingMemoryDraft] = []
        for item in data:
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

    @staticmethod
    def _format_conversation(messages: list[dict]) -> str:
        """Lossless source, including nested tool calls and structured content."""
        return json.dumps(messages, ensure_ascii=False)
