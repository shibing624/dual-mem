# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Extractor that pulls identity/facts and basic profile from a conversation in a
single JSON-mode LLM call, then applies the L0 basic-profile update deterministically.
"""
from dual_mem.agent import prompts
from dual_mem.agent.basic_profile import BasicProfileTool
from dual_mem.providers.llm import LLMClient


class Extractor:
    """Extracts identity/fact memories and applies the L0 basic profile in one LLM call."""

    def __init__(self, *, llm: LLMClient, basic_profile_tool: BasicProfileTool):
        self.llm = llm
        self.basic_profile_tool = basic_profile_tool

    def extract(
        self,
        *,
        content: str,
        current_time: str,
        app_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> dict:
        """Return extracted identity/facts plus any L0 node id created from basic_info."""
        system = prompts.pick(prompts.EXTRACT_ZH, prompts.EXTRACT_EN, content).format(
            content=content, current_time=current_time
        )
        parsed = self.llm.chat_json(system=system, user=content)
        if not isinstance(parsed, dict):
            parsed = {}

        identity = parsed.get("identity") or []
        facts = parsed.get("facts") or []
        basic_info = parsed.get("basic_info")

        l0_node_id = None
        if isinstance(basic_info, dict) and basic_info:
            l0_node_id = self.basic_profile_tool.apply(
                arguments=basic_info,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )

        return {"identity": identity, "facts": facts, "l0_node_id": l0_node_id}
