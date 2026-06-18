import pytest

from dual_mem.agent.basic_profile import BasicProfileTool
from dual_mem.agent.extractor import Extractor
from dual_mem.storage.vector_store import ChromaVectorStore

from conftest import FakeLLMClient


@pytest.fixture
def store(tmp_storage):
    return ChromaVectorStore(tmp_storage)


def test_extract_identity_facts_and_tool_call(store, fake_embed):
    extract_response = {
        "content": (
            '{"identity":[{"content":"用户喜欢喝咖啡","speculate":null,"tags":["food"]}],'
            '"facts":[{"content":"用户昨天去了北京","speculate":null,"tags":["travel"]}]}'
        ),
        "tool_calls": [
            {"function": {"name": "update_basic_user_profile", "arguments": '{"name": "张三"}'}}
        ],
    }
    llm = FakeLLMClient(responses={"extract": extract_response})
    tool = BasicProfileTool(vector=store, embed=fake_embed)
    extractor = Extractor(llm=llm, basic_profile_tool=tool)

    out = extractor.extract(
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
    assert out["l0_node_id"] is not None

    l0 = store.get(out["l0_node_id"])
    assert l0.custom["basic_info_kv"] == {"name": "张三"}


def test_extract_no_tool_call(store, fake_embed):
    llm = FakeLLMClient(
        responses={"extract": {"content": '{"identity":[],"facts":[]}', "tool_calls": []}}
    )
    tool = BasicProfileTool(vector=store, embed=fake_embed)
    extractor = Extractor(llm=llm, basic_profile_tool=tool)

    out = extractor.extract(
        content="hi there",
        current_time="",
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
    )
    assert out == {"identity": [], "facts": [], "l0_node_id": None}
