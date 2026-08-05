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


async def test_content_hash_dedup_scoped_by_session(factory, fake_llm):
    """Same content in different sessions must not share dedup cache."""
    factory.settings = Settings(
        mode="system1",
        storage_dir=factory.settings.storage_dir,
        content_hash_dedup=True,
    )
    fake_llm.responses["extract"] = {
        "facts": [{"content": "重复消息", "tags": []}],
        "identity": [],
        "intentions": [],
        "is_ephemeral": False,
    }
    writer = MemoryWriter(factory=factory)
    content = "完全相同的 benchmark 消息"

    r_a = await writer.write(
        content=content,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="session-a",
        request_id="req-a",
    )
    r_b = await writer.write(
        content=content,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="session-b",
        request_id="req-b",
    )

    assert r_a.memory_id != r_b.memory_id
    where_b = build_filter(app_ids=["app"], user_id="u", session_ids=["session-b"])
    assert any(n.node_id == r_b.memory_id for n in factory.vector.get_many(where_b))


async def test_content_hash_dedup_user_scope_hits_across_sessions(factory, fake_llm):
    """content_hash_scope='user' dedups identical content across different sessions."""
    factory.settings = Settings(
        mode="system1",
        storage_dir=factory.settings.storage_dir,
        content_hash_dedup=True,
        content_hash_scope="user",
    )
    fake_llm.responses["extract"] = {
        "facts": [{"content": "重复消息", "tags": []}],
        "identity": [],
        "intentions": [],
        "is_ephemeral": False,
    }
    writer = MemoryWriter(factory=factory)
    content = "完全相同的 benchmark 消息"

    r_a = await writer.write(
        content=content, app_id="app", user_id="u", agent_id="ag",
        session_id="session-a", request_id="req-a",
    )
    r_b = await writer.write(
        content=content, app_id="app", user_id="u", agent_id="ag",
        session_id="session-b", request_id="req-b",
    )
    # user scope ignores session → second write is a cache hit returning the first outcome.
    assert r_a.memory_id == r_b.memory_id
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["type"] == "chat_json"


async def test_system1_write_ephemeral_returns_only_l1(factory, fake_llm):
    """When extractor flags is_ephemeral, no extras are persisted; raw stays ACTIVE."""
    fake_llm.responses["extract"] = {
        "facts": [],
        "identity": [],
        "intentions": [],
        "is_ephemeral": True,
    }
    writer = MemoryWriter(factory=factory)
    result = await writer.write(
        content="今天的会议安排在下午三点开始具体讨论新版本的发布计划",
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="",
        request_id="req-2",
    )
    assert result.commit_passed is False
    assert result.commit_passed is False
    assert result.is_ephemeral is True
    assert result.extra_node_ids == []
    where = build_filter(app_ids=["app"], user_id="u")
    nodes = factory.vector.get_many(where)
    assert len(nodes) == 1
    assert nodes[0].layer is Layer.L1_RAW
    assert nodes[0].status is MemoryStatus.ACTIVE
