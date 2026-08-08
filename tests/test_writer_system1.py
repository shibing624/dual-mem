import pytest

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.config import Settings
from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.sdk_models import CommitResult
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


@pytest.fixture
def dual_factory(tmp_storage, fake_embed, fake_llm):
    return ComponentFactory(
        settings=Settings(mode="dual", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=fake_llm,
    )


async def test_dual_write_shadows_raw_when_extract_yields_layers(dual_factory, fake_llm):
    """Dual write: extractor returns L4 identity → raw L1 turns SHADOW, L4 stays ACTIVE."""
    fake_llm.responses["extract"] = {
        "facts": [],
        "identity": [{"content": "我喜欢喝咖啡", "tags": ["preference"]}],
        "intentions": [],
        "is_ephemeral": False,
    }
    writer = MemoryWriter(factory=dual_factory)
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
    nodes = dual_factory.vector.get_many(where)
    by_id = {n.node_id: n for n in nodes}
    raw = by_id[result.memory_id]
    assert raw.layer is Layer.L1_RAW
    assert raw.status is MemoryStatus.SHADOW
    extras = [by_id[nid] for nid in result.extra_node_ids if nid in by_id]
    assert any(n.layer is Layer.L4_IDENTITY and n.status is MemoryStatus.ACTIVE for n in extras)


async def test_content_hash_dedup_scoped_by_session(dual_factory, fake_llm):
    """Same content in different sessions must not share dedup cache."""
    dual_factory.settings = Settings(
        mode="dual",
        storage_dir=dual_factory.settings.storage_dir,
        content_hash_dedup=True,
    )
    fake_llm.responses["extract"] = {
        "facts": [{"content": "重复消息", "tags": []}],
        "identity": [],
        "intentions": [],
        "is_ephemeral": False,
    }
    writer = MemoryWriter(factory=dual_factory)
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
    assert any(n.node_id == r_b.memory_id for n in dual_factory.vector.get_many(where_b))


async def test_content_hash_dedup_user_scope_hits_across_sessions(dual_factory, fake_llm):
    """content_hash_scope='user' dedups identical content across different sessions."""
    dual_factory.settings = Settings(
        mode="dual",
        storage_dir=dual_factory.settings.storage_dir,
        content_hash_dedup=True,
        content_hash_scope="user",
    )
    fake_llm.responses["extract"] = {
        "facts": [{"content": "重复消息", "tags": []}],
        "identity": [],
        "intentions": [],
        "is_ephemeral": False,
    }
    writer = MemoryWriter(factory=dual_factory)
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


async def test_dual_write_ephemeral_returns_only_l1(dual_factory, fake_llm):
    """When extractor flags is_ephemeral, no extras are persisted; raw stays ACTIVE."""
    fake_llm.responses["extract"] = {
        "facts": [],
        "identity": [],
        "intentions": [],
        "is_ephemeral": True,
    }
    writer = MemoryWriter(factory=dual_factory)
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
    nodes = dual_factory.vector.get_many(where)
    assert len(nodes) == 1
    assert nodes[0].layer is Layer.L1_RAW
    assert nodes[0].status is MemoryStatus.ACTIVE


async def test_dual_can_drop_shadow_raw_vector_when_derived(
    tmp_path, fake_llm, fake_embed
):
    settings = Settings(
        mode="dual",
        storage_dir=str(tmp_path),
        embed_api_key="test-key",
        embed_dim=4,
        skip_l1_vector_when_derived=True,
    )
    factory = ComponentFactory(settings=settings, llm=fake_llm)
    factory._embed = fake_embed
    writer = MemoryWriter(factory=factory)

    result = await writer.write(
        content="我很喜欢用索尼相机拍照记录生活中的美好瞬间和旅行经历",
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="",
        request_id="req-drop-l1",
    )

    assert result.extra_node_ids
    assert factory.vector.get(result.memory_id) is None
    assert factory.vector.get(result.extra_node_ids[0]) is not None


async def test_dual_keeps_raw_vector_when_no_derived_node(
    tmp_path, fake_llm, fake_embed, monkeypatch
):
    settings = Settings(
        mode="dual",
        storage_dir=str(tmp_path),
        embed_api_key="test-key",
        embed_dim=4,
        skip_l1_vector_when_derived=True,
    )
    factory = ComponentFactory(settings=settings, llm=fake_llm)
    factory._embed = fake_embed
    writer = MemoryWriter(factory=factory)

    async def no_derived_nodes(*args, **kwargs):
        return [], CommitResult(passed=False, reason="test"), False

    monkeypatch.setattr(MemAgent, "run", no_derived_nodes)
    result = await writer.write(
        content="ok",
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="",
        request_id="req-keep-l1",
    )

    assert result.extra_node_ids == []
    assert factory.vector.get(result.memory_id) is not None


async def test_system1_derived_get_own_embeddings_and_raw_shadowed(factory, fake_llm):
    fake_llm.responses["extract"] = {
        "facts": [
            {"content": "用户养了一只猫", "tags": ["pet"]},
            {"content": "猫叫 Miso", "tags": ["pet", "name"]},
        ],
        "identity": [{"content": "用户是猫主人", "tags": ["identity"]}],
        "intentions": [],
        "is_ephemeral": False,
    }
    writer = MemoryWriter(factory=factory)

    result = await writer.write(
        content="user: I adopted a cat named Miso.\nassistant: Miso is a lovely name.",
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="session",
        request_id="req-system1-raw",
    )

    where = build_filter(app_ids=["app"], user_id="u")
    nodes = factory.vector.get_many(where)
    assert result.commit_passed is True
    assert len(result.extra_node_ids) == 3
    assert len(nodes) == 4
    by_ids = factory.vector.get_by_ids([node.node_id for node in nodes])
    nodes = list(by_ids.values())
    raw = next(node for node in nodes if node.node_id == result.memory_id)
    derived = [node for node in nodes if node.node_id in result.extra_node_ids]
    assert raw.layer is Layer.L1_RAW
    # v5: derived memories carry their own per-content embeddings, and the raw
    # chunk leaves the ACTIVE recall pool once children exist.
    assert raw.status is MemoryStatus.SHADOW
    assert all(node.status is MemoryStatus.ACTIVE for node in derived)
    assert all(node.embedding is not None for node in derived)
    assert all(node.embedding != raw.embedding for node in derived)
    assert len({tuple(node.embedding) for node in derived}) == len(derived)
    hits = factory.vector.query(
        embedding=factory.embed.embed_sync("Miso"),
        top_k=1,
        where=where,
    )
    assert len(hits) == 1
    assert hits[0].node_id in {node.node_id for node in nodes}
    assert [call["type"] for call in fake_llm.calls] == ["chat_json"]


async def test_system1_keeps_duplicate_chunks(factory, fake_llm):
    factory.settings.content_hash_dedup = True
    writer = MemoryWriter(factory=factory)
    kwargs = {
        "content": "user: repeated but temporally relevant chunk",
        "app_id": "app",
        "user_id": "u",
        "agent_id": "",
        "session_id": "session",
    }

    first = await writer.write(request_id="req-duplicate-1", **kwargs)
    second = await writer.write(request_id="req-duplicate-2", **kwargs)

    nodes = factory.vector.get_many(build_filter(app_ids=["app"], user_id="u"))
    raw_nodes = [node for node in nodes if node.layer is Layer.L1_RAW]
    assert first.memory_id != second.memory_id
    assert {node.node_id for node in raw_nodes} == {first.memory_id, second.memory_id}
    # Each duplicate chunk still runs its own extraction; the second write may
    # add an inline reconcile LLM call on top of the two extractions.
    extract_calls = [
        call
        for call in fake_llm.calls
        if "memory analyst" in call.get("system", "") or "记忆分析专家" in call.get("system", "")
    ]
    assert len(extract_calls) == 2
