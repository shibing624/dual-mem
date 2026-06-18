# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: OpenAI-compatible LLM client wrapper offering JSON/text/tool-call chat
helpers, plus language detection and tolerant JSON parsing of model output.
"""
import itertools
import json
import logging
import re
import time

from openai import OpenAI

_CJK = re.compile(r"[\u4e00-\u9fff]")

logger = logging.getLogger("dual_mem.llm")

# Process-wide monotonically increasing counter so every real LLM request gets a
# stable sequence number in the logs (helps eyeball "how many calls did one add cost").
_call_seq = itertools.count(1)


def is_chinese(text: str) -> bool:
    """Heuristically decide whether text is Chinese by CJK character ratio."""
    if not text:
        return False
    chinese = len(_CJK.findall(text))
    return chinese / len(text) > 0.1


def _parse_json(content: str):
    """Parse JSON from model output, stripping code fences and extracting the first object/array."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


class LLMClient:
    """Thin synchronous wrapper over an OpenAI-compatible chat completions endpoint."""

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _log_call(self, kind: str, elapsed_ms: float) -> None:
        """Emit an INFO log per real LLM request with a global sequence number."""
        seq = next(_call_seq)
        logger.info(
            "LLM call #%d kind=%s model=%s took=%.0fms", seq, kind, self.model, elapsed_ms
        )

    def chat_json(self, *, system: str, user: str, temperature: float = 0.2) -> dict:
        """Run a chat completion and parse the reply as JSON."""
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        self._log_call("chat_json", (time.perf_counter() - start) * 1000)
        content = resp.choices[0].message.content or ""
        return _parse_json(content)

    def chat_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        """Run a chat completion and return the raw text reply."""
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        self._log_call("chat_text", (time.perf_counter() - start) * 1000)
        return resp.choices[0].message.content or ""

    def chat_with_tools(
        self, *, system: str, user: str, tools: list, temperature: float = 0.2
    ) -> dict:
        """Run a tool-enabled chat completion and return content plus normalized tool calls."""
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=tools,
            temperature=temperature,
        )
        self._log_call("chat_with_tools", (time.perf_counter() - start) * 1000)
        msg = resp.choices[0].message
        tool_calls = []
        for tc in msg.tool_calls or []:
            tool_calls.append(
                {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            )
        return {"content": msg.content or "", "tool_calls": tool_calls}
