import pytest

from dual_mem.config import Settings
from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryStatus
from dual_mem.writer.memory_writer import MemoryWriter

from conftest import FakeLLMClient

EXTRACT_RESPONSE = {
    "content": (
        '{"identity":[{"content":"用户喜欢喝咖啡","speculate":null,"tags":["food"]}],'
        '"facts":[{"content":"用户昨天去了北京","speculate":null,"tags":["travel"]}]}'
    ),
    "tool_calls": [],
}


def _factory(tmp_storage, fake_embed, responses):
    return ComponentFactory(
        settings=Settings(mode="pro", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses=responses),
    )


async def test_pro_write_shadows_raw_and_creates_layers(tmp_storage, fake_embed):
    factory = _factory(tmp_storage, fake_embed, {"extract": EXTRACT_RESPONSE, "search_query": []})
    writer = MemoryWriter(factory=factory, agent_mode="full")

    result = await writer.write(
        content="用户喜欢喝咖啡，昨天去了北京",
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req-1",
    )

    raw = factory.vector.get(result.memory_id)
    assert raw.layer is Layer.L1_RAW
    assert raw.status is MemoryStatus.SHADOW

    assert len(result.extra_node_ids) == 2
    where = build_filter(
        app_ids=["app"],
        user_id="u",
        layers=[Layer.L2_FACT, Layer.L4_IDENTITY],
        statuses=[MemoryStatus.ACTIVE],
    )
    cognitive = factory.vector.get_many(where)
    assert {n.layer for n in cognitive} == {Layer.L2_FACT, Layer.L4_IDENTITY}


async def test_pro_write_long_content_adds_summary(tmp_storage, fake_embed):
    long_text = "用户" + "聊了很多关于旅行和美食的事情。" * 60
    factory = _factory(
        tmp_storage,
        fake_embed,
        {"extract": EXTRACT_RESPONSE, "search_query": [], "text": "用户喜欢旅行和美食。"},
    )
    writer = MemoryWriter(factory=factory, agent_mode="full")

    result = await writer.write(
        content=long_text,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req-2",
    )

    assert len(result.extra_node_ids) == 3
    where = build_filter(app_ids=["app"], user_id="u", layers=[Layer.L3_SUMMARY])
    summaries = factory.vector.get_many(where)
    assert len(summaries) == 1
    assert summaries[0].content == "用户喜欢旅行和美食。"
