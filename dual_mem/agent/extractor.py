# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async extractor that pulls identity/facts/intentions/emotion/basic_info and
is_ephemeral signal from a conversation in a single JSON-mode LLM call, then applies the L0
basic-profile update deterministically.
"""
import logging

from dual_mem.agent import prompts
from dual_mem.agent.basic_profile import BasicProfileTool
from dual_mem.providers.llm import LLMClient

logger = logging.getLogger("dual_mem.agent.extract")


class Extractor:
    """Extracts identity/fact/intention memories and applies the L0 basic profile in one LLM call."""

    def __init__(self, *, llm: LLMClient, basic_profile_tool: BasicProfileTool):
        self.llm = llm
        self.basic_profile_tool = basic_profile_tool

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
        """Return extracted identity/facts/intentions/emotion plus any L0 node id created."""
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

        l0_node_id = None
        if isinstance(basic_info, dict) and basic_info:
            l0_node_id = await self.basic_profile_tool.apply(
                arguments=basic_info,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )

        logger.debug(
            "extract identity=%d facts=%d intentions=%d ephemeral=%s l0=%s",
            len(identity) if isinstance(identity, list) else 0,
            len(facts) if isinstance(facts, list) else 0,
            len(intentions) if isinstance(intentions, list) else 0,
            is_ephemeral,
            bool(l0_node_id),
        )

        return {
            "identity": identity if isinstance(identity, list) else [],
            "facts": facts if isinstance(facts, list) else [],
            "intentions": intentions if isinstance(intentions, list) else [],
            "emotion": emotion if isinstance(emotion, dict) else {},
            "is_ephemeral": is_ephemeral,
            "l0_node_id": l0_node_id,
        }
