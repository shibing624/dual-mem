# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async OpenAI-compatible LLM client wrapper offering JSON (optionally JSON mode)
and text chat helpers, plus language detection, tolerant JSON parsing and per-call INFO logging.
"""
import itertools
import json
import logging
import re
import time

from openai import AsyncOpenAI

_CJK = re.compile(r"[一-鿿]")

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
    """Async wrapper over an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 60,
        json_mode: bool = True,
    ):
        self.model = model
        self.json_mode = json_mode
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _log_call(self, kind: str, elapsed_ms: float) -> None:
        """Emit an INFO log per real LLM request with a global sequence number."""
        seq = next(_call_seq)
        logger.info(
            "LLM call #%d kind=%s model=%s took=%.0fms", seq, kind, self.model, elapsed_ms
        )

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_object: bool | None = None,
    ):
        """Run a chat completion and parse the reply as JSON.

        json_object overrides the client default: True forces OpenAI JSON mode (object only),
        None uses self.json_mode. Callers expecting a top-level array pass json_object=False.
        """
        use_json_mode = self.json_mode if json_object is None else json_object
        kwargs: dict = {}
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **kwargs,
        )
        self._log_call("chat_json", (time.perf_counter() - start) * 1000)
        content = resp.choices[0].message.content or ""
        return _parse_json(content)

    async def chat_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        """Run a chat completion and return the raw text reply."""
        start = time.perf_counter()
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        self._log_call("chat_text", (time.perf_counter() - start) * 1000)
        return resp.choices[0].message.content or ""

    async def chat_with_tools(
        self,
        *,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float = 0.2,
    ) -> dict:
        """Run a tool-calling chat completion; return ``{content, tool_calls}`` from one turn.

        Used by the System2 ReAct loop: caller maintains the messages list, appends the
        assistant turn (with tool_calls) plus role=tool replies, and re-invokes until the
        model emits no more tool_calls.
        """
        start = time.perf_counter()
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            tool_choice=tool_choice,  # type: ignore[arg-type]
            temperature=temperature,
        )
        self._log_call("chat_tools", (time.perf_counter() - start) * 1000)
        msg = resp.choices[0].message
        tool_calls: list[dict] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": fn.name,
                            "arguments": fn.arguments or "",
                        },
                    }
                )
        return {"content": msg.content or "", "tool_calls": tool_calls}
