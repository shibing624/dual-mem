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
    extractor = Extractor(llm=llm)

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
