# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async extractor that pulls identity/facts/intentions/basic_info and
the is_ephemeral commit signal from a conversation in one JSON-mode LLM call. L0 persistence
is deferred to MemAgent so L0/L2/L4 embeddings can be batched post-extract.
"""
import logging

from dual_mem.agent import prompts
from dual_mem.providers.llm import LLMClient, merge_extract_results, truncate_middle

logger = logging.getLogger("dual_mem.agent.extract")

_EXTRACT_RETRY_MIN_CONTENT_LEN = 200


class Extractor:
    """Extracts identity/fact/intention memories from one LLM JSON call."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        max_content_chars: int = 0,
        retry_on_failure: bool = True,
        few_shot_enabled: bool = False,
    ):
        self.llm = llm
        self.max_content_chars = max_content_chars
        self.retry_on_failure = retry_on_failure
        self.few_shot_enabled = few_shot_enabled

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
        """Return extracted fields; ``basic_info`` is persisted later by MemAgent."""
        tmpl = prompts.pick(prompts.EXTRACT_ZH, prompts.EXTRACT_EN, content)
        if self.few_shot_enabled:
            tmpl += prompts.pick(
                prompts.EXTRACT_FEW_SHOT_ZH,
                prompts.EXTRACT_FEW_SHOT_EN,
                content,
            )
        def _build_system(chunk: str) -> str:
            return tmpl.format(
                content=chunk,
                current_time=current_time,
            )

        llm_content = self._prepare_content(content)
        parsed = await self.llm.chat_json_for_content(
            content=llm_content,
            build_system=_build_system,
            merge_results=merge_extract_results,
        )
        if self.retry_on_failure and self._should_retry(parsed, content_len=len(llm_content)):
            logger.warning(
                "extract: retry after empty/unparseable LLM output (content len=%d)",
                len(content),
            )
            retry_tmpl = tmpl + prompts.pick(
                prompts.EXTRACT_RETRY_APPEND_ZH,
                prompts.EXTRACT_RETRY_APPEND_EN,
                content,
            )

            def _build_system_retry(chunk: str) -> str:
                return retry_tmpl.format(
                    content=chunk,
                    current_time=current_time,
                )

            parsed = await self.llm.chat_json_for_content(
                content=llm_content,
                build_system=_build_system_retry,
                merge_results=merge_extract_results,
                temperature=0.0,
            )

        if not isinstance(parsed, dict) or not parsed:
            logger.warning(
                "extract: empty/unparseable LLM output for content len=%d (preview=%r)",
                len(content),
                content[:120],
            )
            parsed = {} if not isinstance(parsed, dict) else parsed

        identity = parsed.get("identity") or []
        facts = parsed.get("facts") or []
        intentions = parsed.get("intentions") or []
        is_ephemeral = bool(parsed.get("is_ephemeral", False))
        basic_info = parsed.get("basic_info")
        if not isinstance(basic_info, dict):
            basic_info = {}

        logger.debug(
            "extract identity=%d facts=%d intentions=%d ephemeral=%s basic_info=%s",
            len(identity) if isinstance(identity, list) else 0,
            len(facts) if isinstance(facts, list) else 0,
            len(intentions) if isinstance(intentions, list) else 0,
            is_ephemeral,
            bool(basic_info),
        )

        out: dict = {
            "identity": identity if isinstance(identity, list) else [],
            "facts": facts if isinstance(facts, list) else [],
            "intentions": intentions if isinstance(intentions, list) else [],
            "is_ephemeral": is_ephemeral,
            "basic_info": basic_info,
        }
        return out

    def _prepare_content(self, content: str) -> str:
        if self.max_content_chars <= 0 or len(content) <= self.max_content_chars:
            return content
        logger.info(
            "extract: truncating content %d -> %d chars",
            len(content),
            self.max_content_chars,
        )
        return truncate_middle(content, self.max_content_chars)

    @staticmethod
    def _should_retry(parsed: dict | list | None, *, content_len: int) -> bool:
        if content_len < _EXTRACT_RETRY_MIN_CONTENT_LEN:
            return False
        if not isinstance(parsed, dict):
            return True
        return not parsed
