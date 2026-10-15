"""Text-generation prompts. Jev owns persistence and relation decisions."""

from dual_mem.providers.llm import is_chinese

EXTRACT_EN = """Extract explicit atomic statements from the source below. Generate text only:
Jev independently decides persistence value, known versus update, representation and relations.
Do not infer implicit preferences, personality or facts from behavior. Do not discard a
statement because it is temporary, repeated or already known. Separate state from distinct
events and independent dimensions. Preserve every name, number, condition, date and attribution.
Assistant claims must remain attributed to the assistant unless confirmed by the user.
Use the source language. Resolve relative times only when the current time is sufficient;
otherwise preserve the ambiguity. Never turn fiction, a question or a hypothesis into a fact.
Source:
---
{content}
---
Current time: {current_time}
Return JSON only, with exactly these fields:
{{"basic_info": {{}}, "identity": [], "intentions": [],
  "facts": [{{"content": "a self-contained explicit statement", "tags": []}}],
  "is_ephemeral": false}}
Put every supported explicit candidate in facts. Do not populate inferred identity.
An empty facts list is allowed only when there is no explicit statement to represent.
"""

EXTRACT_ZH = """只抽取来源中明确陈述的原子候选，负责生成文本。
持久价值、已知还是更新、表示类别和更新关系全部由 Jev 独立决定。
不得根据行为推断隐式偏好、性格或未陈述的事实。不得因临时、重复或已知而丢弃候选。
当前状态和独立事件分开，不同维度分开；保留全部姓名、数字、条件、日期和来源归属。
助手提供的主张保持助手归属，除非用户明确确认。使用来源语言。只有当前时间足够明确
时才能转换相对时间，否则保留不确定性。不得把虚构、问题或假设改写成真实事实。
来源：
---
{content}
---
当前时间：{current_time}
只返回 JSON，字段固定：
{{"basic_info": {{}}, "identity": [], "intentions": [],
  "facts": [{{"content": "明确且自包含的原子陈述", "tags": []}}],
  "is_ephemeral": false}}
所有明确且有来源支持的候选都放入 facts；不要生成隐式 identity。
只有来源没有可表示的明确陈述时 facts 才能为空。
"""

EXTRACT_FEW_SHOT_EN = """
Example source: The user moved from Beijing to Shanghai and now lives in Shanghai.
Output: {{"basic_info": {{}}, "identity": [], "intentions": [], "facts": [
{{"content": "The user moved from Beijing to Shanghai.", "tags": []}},
{{"content": "The user currently lives in Shanghai.", "tags": []}}], "is_ephemeral": false}}
"""
EXTRACT_FEW_SHOT_ZH = """
示例来源：用户从北京搬到上海，现在住在上海。
输出：{{"basic_info": {{}}, "identity": [], "intentions": [], "facts": [
{{"content": "用户从北京搬到上海。", "tags": []}},
{{"content": "用户目前住在上海。", "tags": []}}], "is_ephemeral": false}}
"""

EXTRACT_RETRY_APPEND_ZH = """

## 重要：输出格式（重试）

上一次输出无法解析为 JSON。这次**只输出一个 JSON 对象**，不要任何前后说明、不要 Markdown 代码块标记、不要多余文字。第一个字符必须是 `{{`，最后一个字符必须是 `}}`。"""

EXTRACT_RETRY_APPEND_EN = """

## IMPORTANT: output format (retry)

The previous output could not be parsed as JSON. This time output **exactly one JSON object** with no prose before/after, no Markdown code fences. The first character must be `{{` and the last `}}`."""


def pick(zh_prompt: str, en_prompt: str, text: str) -> str:
    """Select the source language for faithful text generation."""
    return zh_prompt if is_chinese(text) else en_prompt
