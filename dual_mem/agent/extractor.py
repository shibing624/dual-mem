# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async extractor that pulls identity/facts/intentions/emotion/basic_info and
is_ephemeral signal from a conversation in a single JSON-mode LLM call. L0 persistence is
deferred to MemAgent so L0/L2/L4 embeddings can be batched post-extract.
"""
import logging

from dual_mem.agent import prompts
from dual_mem.providers.llm import LLMClient

logger = logging.getLogger("dual_mem.agent.extract")


class Extractor:
    """Extracts identity/fact/intention memories from one LLM JSON call."""

    def __init__(self, *, llm: LLMClient):
        self.llm = llm

    async def extract(
        self,
        *,
        content: str,
        current_time: str,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> dict:
        """Return extracted fields; ``basic_info`` is persisted later by MemAgent."""
        system = prompts.pick(prompts.EXTRACT_ZH, prompts.EXTRACT_EN, content).format(
            content=content, current_time=current_time
        )
        parsed = await self.llm.chat_json(system=system, user=content)
        if not isinstance(parsed, dict):
            parsed = {}

        identity = parsed.get("identity") or []
        facts = parsed.get("facts") or []
        intentions = parsed.get("intentions") or []
        emotion = parsed.get("emotion") or {}
        is_ephemeral = bool(parsed.get("is_ephemeral", False))
        basic_info = parsed.get("basic_info")
        if not isinstance(basic_info, dict):
            basic_info = {}

        logger.debug(
            "extract identity=%d facts=%d intentions=%d ephemeral=%s basic_info=%s",
            len(identity) if isinstance(identity, list) else 0,
            len(facts) if isinstance(facts, list) else 0,
            len(intentions) if isinstance(intentions, list) else 0,
            is_ephemeral,
            bool(basic_info),
        )

        return {
            "identity": identity if isinstance(identity, list) else [],
            "facts": facts if isinstance(facts, list) else [],
            "intentions": intentions if isinstance(intentions, list) else [],
            "emotion": emotion if isinstance(emotion, dict) else {},
            "is_ephemeral": is_ephemeral,
            "basic_info": basic_info,
        }
