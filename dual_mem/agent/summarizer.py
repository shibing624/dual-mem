"""Summarizer：为较长对话生成 L3_SUMMARY 文本。短于阈值则不生成。"""

from dual_mem.agent import prompts
from dual_mem.providers.llm import LLMClient

MIN_CONTENT_LENGTH = 500


class Summarizer:
    def __init__(self, *, llm: LLMClient):
        self.llm = llm

    def summarize(self, *, content: str, current_time: str) -> str | None:
        if len(content) < MIN_CONTENT_LENGTH:
            return None
        system = prompts.pick(prompts.SUMMARY_ZH, prompts.SUMMARY_EN, content).format(
            content=content, current_time=current_time
        )
        summary = self.llm.chat_text(system=system, user=content).strip()
        return summary or None
