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
from json.decoder import JSONDecoder
from typing import Any

from openai import AsyncOpenAI

from dual_mem.providers.usage import UsageCallback, UsageEvent, tokens_from_usage

_CJK = re.compile(r"[一-鿿]")

logger = logging.getLogger("dual_mem.llm")

# Process-wide monotonically increasing counter so every real LLM request gets a
# stable sequence number in the logs (helps eyeball "how many calls did one add cost").
_call_seq = itertools.count(1)

# Default cap for chat_json. Models with json_mode disabled (e.g. Volces ds-v4-flash)
# rely on prose JSON which silently truncates without an explicit max_tokens.
DEFAULT_CHAT_JSON_MAX_TOKENS = 4096


def is_chinese(text: str) -> bool:
    """Heuristically decide whether text is Chinese by CJK character ratio."""
    if not text:
        return False
    chinese = len(_CJK.findall(text))
    return chinese / len(text) > 0.1


def _parse_json(content: str):
    """Parse JSON from model output, with partial-recovery fallbacks.

    Recovery order:
      1. strip ``` fences and try strict json.loads
      2. greedy match {…} or […] and try again
      3. for truncated extractor responses, recover string list values for top-level
         keys (facts / identity / intentions / memories) by scanning closed strings
         before the parse breaks.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            text = match.group(1)

    recovered = _recover_partial_object(text)
    if recovered is not None:
        return recovered
    # last resort: empty dict so downstream code (extractor) treats as "nothing extracted"
    # while the LLM call counts as a failure for telemetry. Caller may still log raw output.
    logger.warning("chat_json output unparseable, returning empty object (len=%d)", len(content))
    return {}


_RECOVERABLE_KEYS = ("facts", "identity", "intentions", "memories", "updates", "ops")


def _recover_partial_object(text: str) -> dict | None:
    """Recover string list values for known keys from truncated JSON.

    Returns ``{key: [str, ...]}`` for any key in _RECOVERABLE_KEYS whose value array
    we can salvage (closed strings before the truncation point). Returns None if
    no key matched at all — caller decides whether to fall back to empty dict.
    """
    out: dict = {}
    for key in _RECOVERABLE_KEYS:
        m = re.search(rf'"{key}"\s*:\s*\[', text)
        if not m:
            continue
        rest = text[m.end():]
        decoder = JSONDecoder()
        idx = 0
        items: list[str] = []
        while idx < len(rest):
            while idx < len(rest) and rest[idx] in " \t\n\r,":
                idx += 1
            if idx >= len(rest) or rest[idx] == "]":
                break
            if rest[idx] != '"':
                break
            try:
                value, end = decoder.raw_decode(rest[idx:])
            except json.JSONDecodeError:
                break
            if isinstance(value, str) and value.strip():
                items.append(value.strip())
            idx += end
        if items:
            out[key] = items
    return out or None


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
        extra_body: dict[str, Any] | None = None,
        usage_callback: UsageCallback | None = None,
    ):
        self.model = model
        self.json_mode = json_mode
        self.extra_body = extra_body or {}
        self.usage_callback = usage_callback
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _completion_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Merge caller kwargs with configured extra_body (e.g. thinking depth)."""
        out = dict(kwargs)
        if self.extra_body:
            out["extra_body"] = self.extra_body
        return out

    def _log_call(self, kind: str, elapsed_ms: float) -> None:
        """Emit an INFO log per real LLM request with a global sequence number."""
        seq = next(_call_seq)
        logger.info(
            "LLM call #%d kind=%s model=%s took=%.0fms", seq, kind, self.model, elapsed_ms
        )

    def _after_call(self, kind: str, resp: Any, start: float) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._log_call(kind, elapsed_ms)
        if self.usage_callback is None:
            return
        prompt_tokens, completion_tokens = tokens_from_usage(resp)
        self.usage_callback(
            UsageEvent(
                kind=kind,
                model=self.model,
                latency_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_object: bool | None = None,
        max_tokens: int = DEFAULT_CHAT_JSON_MAX_TOKENS,
    ):
        """Run a chat completion and parse the reply as JSON.

        json_object overrides the client default: True forces OpenAI JSON mode (object only),
        None uses self.json_mode. Callers expecting a top-level array pass json_object=False.
        max_tokens is required to avoid silent truncation of large extractor outputs on
        providers that do not support response_format=json_object.
        """
        use_json_mode = self.json_mode if json_object is None else json_object
        kwargs: dict = {"max_tokens": max_tokens}
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
            **self._completion_kwargs(**kwargs),
        )
        self._after_call("chat_json", resp, start)
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
            **self._completion_kwargs(),
        )
        self._after_call("chat_text", resp, start)
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
            **self._completion_kwargs(),
        )
        self._after_call("chat_tools", resp, start)
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
