import httpx
import pytest
import respx

from dual_mem.providers.llm import LLMClient, is_chinese


def _make_client():
    return LLMClient(base_url="https://api.test/v1", api_key="sk-x", model="gpt-test")


def _completion(message: dict) -> dict:
    return {
        "id": "1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


@respx.mock
def test_chat_json_parses_fenced_content():
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json=_completion(
                {"role": "assistant", "content": '```json\n{"facts": ["a"]}\n```'}
            ),
        )
    )
    result = _make_client().chat_json(system="s", user="u")
    assert result == {"facts": ["a"]}
    assert route.called


@respx.mock
def test_chat_json_regex_fallback():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json=_completion(
                {"role": "assistant", "content": 'noise before {"k": 1} trailing'}
            ),
        )
    )
    result = _make_client().chat_json(system="s", user="u")
    assert result == {"k": 1}


@respx.mock
def test_chat_with_tools_returns_tool_calls():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json=_completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"x":1}'},
                        }
                    ],
                }
            ),
        )
    )
    tools = [{"type": "function", "function": {"name": "add", "parameters": {}}}]
    result = _make_client().chat_with_tools(system="s", user="u", tools=tools)
    assert result["content"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "add"
    assert result["tool_calls"][0]["function"]["arguments"] == '{"x":1}'


@pytest.mark.parametrize(
    "text,expected",
    [
        ("你好，世界，这是一段中文文本", True),
        ("This is purely English text only", False),
        ("", False),
        ("hello 你好 world test more english here", False),
    ],
)
def test_is_chinese(text, expected):
    assert is_chinese(text) is expected
