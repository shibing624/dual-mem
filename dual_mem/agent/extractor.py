"""Extractor：单次 LLM 调用，从对话提取 identity / facts，并通过 function-calling
工具更新 L0 basic profile。

与源码的差异（M3 简化）：源码用多轮 tool-loop（执行 tool 后把结果回灌 LLM 再出最终
JSON）。我们的 LLMClient 是单轮 system+user，无 tool_call_id 回灌能力，因此采用单轮：
一次 chat_with_tools 同时拿 content(JSON identity/facts) + tool_calls，分别处理。
"""

import json
import re

from dual_mem.agent import prompts
from dual_mem.agent.basic_profile import TOOL_NAME, BasicProfileTool, openai_tool_schema
from dual_mem.providers.llm import LLMClient


def _parse_content_json(text: str) -> dict:
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
        system = prompts.pick(prompts.EXTRACT_ZH, prompts.EXTRACT_EN, content).format(
            content=content, current_time=current_time
        )
        result = self.llm.chat_with_tools(
            system=system, user=content, tools=[openai_tool_schema()]
        )

        parsed = _parse_content_json(result.get("content", ""))
        identity = parsed.get("identity") or []
        facts = parsed.get("facts") or []

        l0_node_id = None
        for tc in result.get("tool_calls") or []:
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
