# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async summarizer that produces L3_SUMMARY text for long conversations,
skipping inputs shorter than a minimum length threshold.
"""
import logging

from dual_mem.agent import prompts
from dual_mem.providers.llm import LLMClient

logger = logging.getLogger("dual_mem.agent.summarize")

MIN_CONTENT_LENGTH = 500


class Summarizer:
    """Generates a concise summary for sufficiently long content."""

    def __init__(self, *, llm: LLMClient):
        self.llm = llm

    async def summarize(self, *, content: str, current_time: str) -> str | None:
        """Summarize content into one short paragraph, or None if below the length threshold."""
        if len(content) < MIN_CONTENT_LENGTH:
            return None
        system = prompts.pick(prompts.SUMMARY_ZH, prompts.SUMMARY_EN, content).format(
            content=content, current_time=current_time
        )
        summary = (await self.llm.chat_text(system=system, user=content)).strip()
        logger.debug("summarize content_len=%d summary_len=%d", len(content), len(summary))
        return summary or None
