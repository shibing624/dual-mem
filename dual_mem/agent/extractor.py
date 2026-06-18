# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Extractor that pulls identity/facts from a conversation and updates the L0
basic profile via function-calling, falling back to a JSON-only call when needed.
"""
import json
import re

from dual_mem.agent import prompts
from dual_mem.agent.basic_profile import TOOL_NAME, BasicProfileTool, openai_tool_schema
from dual_mem.providers.llm import LLMClient


def _parse_content_json(text: str) -> dict:
    """Parse a JSON object from model content, tolerating code fences and surrounding text."""
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}
    return obj if isinstance(obj, dict) else {}


class Extractor:
    """Extracts identity/fact memories and applies L0 basic-profile tool calls."""

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
        """Return extracted identity/facts plus any L0 node id created by the profile tool."""
        system = prompts.pick(prompts.EXTRACT_ZH, prompts.EXTRACT_EN, content).format(
            content=content, current_time=current_time
        )
        result = self.llm.chat_with_tools(
            system=system, user=content, tools=[openai_tool_schema()]
        )

        parsed = _parse_content_json(result.get("content", ""))
        tool_calls = result.get("tool_calls") or []

        if not parsed:
            reparsed = self.llm.chat_json(system=system, user=content)
            parsed = reparsed if isinstance(reparsed, dict) else {}

        identity = parsed.get("identity") or []
        facts = parsed.get("facts") or []

        l0_node_id = None
        for tc in tool_calls:
            fn = tc.get("function") or {}
            if fn.get("name") != TOOL_NAME:
                continue
            raw_args = fn.get("arguments")
            if isinstance(raw_args, str):
                raw_args = json.loads(raw_args) if raw_args.strip() else {}
            elif not isinstance(raw_args, dict):
                raw_args = {}
            nid = self.basic_profile_tool.apply(
                arguments=raw_args,
                app_id=app_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )
            if nid and l0_node_id is None:
                l0_node_id = nid

        return {"identity": identity, "facts": facts, "l0_node_id": l0_node_id}
