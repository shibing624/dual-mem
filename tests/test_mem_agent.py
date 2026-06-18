import pytest

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient

EXTRACT_RESPONSE = {
    "content": (
        '{"identity":[{"content":"用户喜欢喝咖啡","speculate":null,"tags":["food"]}],'
        '"facts":[{"content":"用户昨天去了北京","speculate":null,"tags":["travel"]}]}'
    ),
    "tool_calls": [],
}


def _factory(tmp_storage, fake_embed, responses):
    factory = ComponentFactory(
        settings=Settings(mode="pro", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses=responses),
    )
    return factory


def _raw(fake_embed, content):
    node = MemoryNode(content=content, layer=Layer.L1_RAW, app_id="app", user_id="u", agent_id="ag")
    node.embedding = fake_embed.embed(content)
    return node


def test_run_produces_l2_l4(tmp_storage, fake_embed):
    factory = _factory(tmp_storage, fake_embed, {"extract": EXTRACT_RESPONSE, "search_query": []})
    agent = MemAgent(factory=factory)
    raw = _raw(fake_embed, "用户喜欢喝咖啡，昨天去了北京")
    factory.vector.upsert([raw])

    stored_ids = agent.run(
        raw_node=raw,
        content="用户喜欢喝咖啡，昨天去了北京",
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req-1",
        memory_at=None,
    )

    assert len(stored_ids) == 2
    layers = {factory.vector.get(nid).layer for nid in stored_ids}
    assert layers == {Layer.L4_IDENTITY, Layer.L2_FACT}
    for nid in stored_ids:
        node = factory.vector.get(nid)
        assert node.status is MemoryStatus.ACTIVE
        assert node.is_latest is True


def test_run_long_content_adds_l3(tmp_storage, fake_embed):
    long_text = "用户" + "聊了很多关于旅行和美食的事情。" * 60
    assert len(long_text) >= 500
    factory = _factory(
        tmp_storage,
        fake_embed,
        {"extract": EXTRACT_RESPONSE, "search_query": [], "text": "用户喜欢旅行和美食。"},
    )
    agent = MemAgent(factory=factory)
    raw = _raw(fake_embed, long_text)

    stored_ids = agent.run(
        raw_node=raw,
        content=long_text,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req-2",
        memory_at=None,
    )

    layers = [factory.vector.get(nid).layer for nid in stored_ids]
    assert Layer.L3_SUMMARY in layers
    assert len(stored_ids) == 3
