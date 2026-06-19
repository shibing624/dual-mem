import pytest

from dual_mem.agent.basic_profile import BasicProfileTool
from dual_mem.agent.extractor import Extractor
from dual_mem.storage.vector_store import ChromaVectorStore

from conftest import FakeLLMClient


@pytest.fixture
def store(tmp_storage):
    return ChromaVectorStore(tmp_storage)


async def test_extract_identity_facts_and_basic_info(store, fake_embed):
    extract_response = {
        "identity": [{"content": "用户喜欢喝咖啡", "speculate": None, "tags": ["food"]}],
        "facts": [{"content": "用户昨天去了北京", "speculate": None, "tags": ["travel"]}],
        "basic_info": {"name": "张三"},
    }
    llm = FakeLLMClient(responses={"extract": extract_response})
    tool = BasicProfileTool(vector=store, embed=fake_embed)
    extractor = Extractor(llm=llm, basic_profile_tool=tool)

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
    assert out["l0_node_id"] is not None

    l0 = store.get(out["l0_node_id"])
    assert l0.custom["basic_info_kv"] == {"name": "张三"}

    # 只调用一次 LLM（合并后的 Extractor 不再有 chat_with_tools + fallback 两次）。
    assert sum(1 for c in llm.calls if c["type"] == "chat_json") == 1


async def test_extract_single_call_no_basic_info(store, fake_embed):
    llm = FakeLLMClient(
        responses={"extract": {"identity": [], "facts": [], "intentions": [], "basic_info": {}}}
    )
    tool = BasicProfileTool(vector=store, embed=fake_embed)
    extractor = Extractor(llm=llm, basic_profile_tool=tool)

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
    assert out["l0_node_id"] is None
