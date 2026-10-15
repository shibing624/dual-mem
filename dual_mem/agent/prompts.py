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

SUMMARY_ZH = """为以下对话内容生成简洁的摘要。

内容:
---
{content}
---

当前时间: {current_time}

要求:
1. **第三人称**: 以"用户..."描述用户——不要使用没有明确先行词的代词。
2. **长度**: 1-3 句话，最多 200 字。
3. **优先级（内容超出长度限制时）**:
   a) 变化、决定、承诺（最高）
   b) 明确的偏好、态度、喜恶
   c) 关键事件和事实
   d) 背景信息（最低）
4. **保留偏好信号**: 保留任何直接或间接表达的喜好、厌恶、态度或观点。
5. **自包含**: 读者应该在不看原始对话的情况下就能理解摘要。
6. **不要编造**: 不要添加原始内容中没有的信息。
7. **语言**: 输出语言必须与输入语言一致。
8. **时间**:
   - 如果提供了 `当前时间`，将相对时间表达转换为绝对引用。
   - 如果 `当前时间` 为空，重写句子使其不包含时间。

## 输出约定

1. 只输出摘要文本——一段 1-3 句话的纯文本。
2. 不要用引号、反引号、代码块或任何 markdown 包裹输出。
3. 不要添加前缀或标签（不要"摘要："、"Summary："、"这是..."等）。
4. 不要添加尾部解释或元评论。

现在生成摘要。"""

SUMMARY_EN = """Generate a concise summary of the following conversation content.

Content:
---
{content}
---

Current time: {current_time}

Requirements:
1. **Third-person voice**: Describe the user as "The user ...".
2. **Length**: 1-3 sentences, max 200 words.
3. **Priorities (when content exceeds the length budget)**:
   a) Changes, decisions, commitments (highest)
   b) Explicit preferences, attitudes, dislikes/likes
   c) Key events and facts
   d) Background context (lowest)
4. **Preserve preference signals**.
5. **Self-contained**.
6. **No fabrication**.
7. **Language**: must match the input.
8. **Time**:
   - With `current_time`, resolve relative expressions to absolute references.
   - Without it, rewrite atemporally.

## Output contract

1. Output the summary text ONLY — one paragraph of 1-3 sentences, plain prose.
2. Do NOT wrap in quotes, backticks, code fences, or markdown.
3. Do NOT add a prefix or label.
4. Do NOT add trailing explanations or meta-commentary.

Output the summary now."""


def pick(zh_prompt: str, en_prompt: str, text: str) -> str:
    """Select the source language for faithful text generation."""
    return zh_prompt if is_chinese(text) else en_prompt
