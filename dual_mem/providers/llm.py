import json
import re

from openai import OpenAI

_CJK = re.compile(r"[\u4e00-\u9fff]")


def is_chinese(text: str) -> bool:
    if not text:
        return False
    chinese = len(_CJK.findall(text))
    return chinese / len(text) > 0.1


def _parse_json(content: str):
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
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat_json(self, *, system: str, user: str, temperature: float = 0.2) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        content = resp.choices[0].message.content or ""
        return _parse_json(content)

    def chat_with_tools(
        self, *, system: str, user: str, tools: list, temperature: float = 0.2
    ) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=tools,
            temperature=temperature,
        )
        msg = resp.choices[0].message
        tool_calls = []
        for tc in msg.tool_calls or []:
            tool_calls.append(
                {"function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            )
        return {"content": msg.content or "", "tool_calls": tool_calls}
