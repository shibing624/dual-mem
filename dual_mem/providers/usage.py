# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Optional per-call usage telemetry for LLM and embedding providers.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UsageEvent:
    """One completed LLM or embedding API call."""

    kind: str
    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    text_chars: int = 0
    batch_size: int = 0


UsageCallback = Callable[[UsageEvent], None]


def tokens_from_usage(resp: Any) -> tuple[int, int]:
    """Read prompt/completion token counts from an OpenAI chat completion response.
    
    Args:
        resp: OpenAI chat completion response

    Returns:
        tuple[int, int]: prompt_tokens, completion_tokens
    """
    usage = resp.usage
    if usage is None:
        return 0, 0
    return int(usage.prompt_tokens or 0), int(usage.completion_tokens or 0)
