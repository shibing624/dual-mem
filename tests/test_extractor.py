import pytest

from dual_mem.agent.extractor import Extractor

from conftest import FakeLLMClient


async def test_extract_identity_facts_and_basic_info():
    extract_response = {
        "identity": [{"content": "用户喜欢喝咖啡", "speculate": None, "tags": ["food"]}],
        "facts": [{"content": "用户昨天去了北京", "speculate": None, "tags": ["travel"]}],
        "basic_info": {"name": "张三"},
    }
    llm = FakeLLMClient(responses={"extract": extract_response})
    extractor = Extractor(llm=llm)

    out = await extractor.extract(
        content="用户喜欢喝咖啡，昨天去了北京，我叫张三",
        current_time="",
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
    )

    assert len(out["identity"]) == 1
    assert out["identity"][0]["content"] == "用户喜欢喝咖啡"
    assert len(out["facts"]) == 1
    assert out["facts"][0]["content"] == "用户昨天去了北京"
    assert out["basic_info"] == {"name": "张三"}
    assert sum(1 for c in llm.calls if c["type"] == "chat_json") == 1


async def test_extract_single_call_no_basic_info():
    llm = FakeLLMClient(
        responses={"extract": {"identity": [], "facts": [], "intentions": [], "basic_info": {}}}
    )
    extractor = Extractor(llm=llm, retry_on_failure=False)

    out = await extractor.extract(
        content="hi there",
        current_time="",
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
    )
    assert out["identity"] == []
    assert out["facts"] == []
    assert out["intentions"] == []
    assert out["basic_info"] == {}


async def test_extract_retries_on_empty_json():
    calls = {"n": 0}

    def flaky_extract(*, system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            return {}
        return {
            "identity": [{"content": "用户喜欢咖啡", "speculate": None, "tags": []}],
            "facts": [],
            "intentions": [],
            "basic_info": {},
        }

    llm = FakeLLMClient(responses={"extract": flaky_extract})
    extractor = Extractor(llm=llm, retry_on_failure=True)

    out = await extractor.extract(
        content="x" * 500,
        current_time="",
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
    )

    assert len(out["identity"]) == 1
    assert sum(1 for c in llm.calls if c["type"] == "chat_json") == 2
    # Retry runs at temperature=0 with a JSON-only reinforcement appended to the system prompt.
    assert llm.calls[1]["kw"]["temperature"] == 0.0
    assert "json_object" not in llm.calls[1]["kw"]
    assert "JSON" in llm.calls[1]["system"]
    assert llm.calls[1]["system"] != llm.calls[0]["system"]


async def test_extract_truncates_long_content():
    seen: dict[str, int] = {}

    def capture(*, system, user):
        seen["len"] = len(user)
        return {"identity": [], "facts": [], "intentions": [], "basic_info": {}}

    llm = FakeLLMClient(responses={"extract": capture})
    extractor = Extractor(llm=llm, max_content_chars=10_000, retry_on_failure=False)

    await extractor.extract(
        content="a" * 20_000,
        current_time="",
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
    )

    assert seen["len"] == 10_000
