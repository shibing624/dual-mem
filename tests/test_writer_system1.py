import pytest

from dual_mem.config import Settings
from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryStatus
from dual_mem.writer.memory_writer import MemoryWriter


@pytest.fixture
def factory(tmp_storage, fake_embed, fake_llm):
    f = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=fake_llm,
    )
    return f


async def test_system1_write_shadows_raw_when_extract_yields_layers(factory, fake_llm):
    """system1 write: extractor returns L4 identity → raw L1 turns SHADOW, L4 stays ACTIVE."""
    fake_llm.responses["extract"] = {
        "facts": [],
        "identity": [{"content": "我喜欢喝咖啡", "tags": ["preference"]}],
        "intentions": [],
        "is_ephemeral": False,
    }
    writer = MemoryWriter(factory=factory)
    result = await writer.write(
        content="我喜欢喝咖啡",
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="",
        request_id="req-1",
    )

    assert result.extra_node_ids
    where = build_filter(app_ids=["app"], user_id="u")
    nodes = factory.vector.get_many(where)
    by_id = {n.node_id: n for n in nodes}
    raw = by_id[result.memory_id]
    assert raw.layer is Layer.L1_RAW
    assert raw.status is MemoryStatus.SHADOW
    extras = [by_id[nid] for nid in result.extra_node_ids if nid in by_id]
    assert any(n.layer is Layer.L4_IDENTITY and n.status is MemoryStatus.ACTIVE for n in extras)


async def test_system1_write_ephemeral_returns_only_l1(factory, fake_llm):
    """When extractor flags is_ephemeral, no extras are persisted; raw stays ACTIVE."""
    fake_llm.responses["extract"] = {
        "facts": [],
        "identity": [],
        "intentions": [],
        "is_ephemeral": True,
    }
    # Use a longer informative-looking content so the gate lets it through and the
    # extractor's is_ephemeral verdict is what determines the outcome.
    writer = MemoryWriter(factory=factory)
    result = await writer.write(
        content="今天的会议安排在下午三点开始具体讨论新版本的发布计划",
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="",
        request_id="req-2",
    )
    assert result.gate_passed is True
    assert result.is_ephemeral is True
    assert result.extra_node_ids == []
    where = build_filter(app_ids=["app"], user_id="u")
    nodes = factory.vector.get_many(where)
    assert len(nodes) == 1
    assert nodes[0].layer is Layer.L1_RAW
    assert nodes[0].status is MemoryStatus.ACTIVE
