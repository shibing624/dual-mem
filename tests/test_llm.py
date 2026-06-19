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
async def test_chat_json_parses_fenced_content():
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json=_completion(
                {"role": "assistant", "content": '```json\n{"facts": ["a"]}\n```'}
            ),
        )
    )
    result = await _make_client().chat_json(system="s", user="u")
    assert result == {"facts": ["a"]}
    assert route.called


@respx.mock
async def test_chat_json_regex_fallback():
    respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json=_completion(
                {"role": "assistant", "content": 'noise before {"k": 1} trailing'}
            ),
        )
    )
    result = await _make_client().chat_json(system="s", user="u")
    assert result == {"k": 1}


@respx.mock
async def test_chat_json_sends_json_mode_by_default():
    import json as _json

    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion({"role": "assistant", "content": "{}"}))
    )
    await _make_client().chat_json(system="s", user="u")
    body = _json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}


@respx.mock
async def test_chat_json_disabled_json_mode_omits_response_format():
    import json as _json

    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion({"role": "assistant", "content": "[]"}))
    )
    # client-level json_mode off
    await LLMClient(
        base_url="https://api.test/v1", api_key="sk-x", model="gpt-test", json_mode=False
    ).chat_json(system="s", user="u")
    body = _json.loads(route.calls.last.request.content)
    assert "response_format" not in body

    # per-call override off
    await _make_client().chat_json(system="s", user="u", json_object=False)
    body2 = _json.loads(route.calls.last.request.content)
    assert "response_format" not in body2


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
