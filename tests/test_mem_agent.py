import asyncio

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient

EXTRACT_RESPONSE = {
    "is_ephemeral": False,
    "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
    "identity": [{"content": "用户喜欢喝咖啡", "speculate": None, "tags": ["food"]}],
    "facts": [{"content": "用户昨天去了北京", "speculate": None, "tags": ["travel"]}],
    "intentions": [],
    "basic_info": {},
    "gate_decision": {
        "novelty": 0.9,
        "biographical_relevance": 0.9,
        "emotional_arousal": 0.2,
        "reason": "test",
    },
}


def _factory(tmp_storage, fake_embed, responses):
    factory = ComponentFactory(
        # gate_enabled=False so trivial test inputs aren't filtered out by the heuristic gate.
        settings=Settings(mode="system1", storage_dir=tmp_storage, gate_enabled=False),
        embed=fake_embed,
        llm=FakeLLMClient(responses=responses),
    )
    return factory


def _raw(fake_embed, content):
    node = MemoryNode(content=content, layer=Layer.L1_RAW, app_id="app", user_id="u", agent_id="ag")
    node.embedding = fake_embed.embed_sync(content)
    return node


async def test_combined_gate_extract_one_llm_call(tmp_storage, fake_embed):
    """combined_gate_extract merges gate scoring into the extract JSON call."""
    factory = ComponentFactory(
        settings=Settings(
            mode="system1",
            storage_dir=tmp_storage,
            gate_enabled=True,
            combined_gate_extract=True,
        ),
        embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": EXTRACT_RESPONSE}),
    )
    agent = MemAgent(factory=factory)
    raw = _raw(fake_embed, "用户喜欢喝咖啡，昨天去了北京")
    factory.vector.upsert([raw])

    stored_ids, gate_result, is_ephemeral = await agent.run(
        raw_node=raw,
        content="用户喜欢喝咖啡，昨天去了北京",
        embedding=raw.embedding,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req-combined",
        memory_at=None,
    )

    assert is_ephemeral is False
    assert gate_result.passed is True
    assert len(stored_ids) == 2
    gate_calls = [
        c for c in factory.llm.calls
        if c["type"] == "chat_json"
        and ("记忆价值评估" in c["system"] or "memory value gate" in c["system"])
    ]
    assert gate_calls == []


async def test_combined_gate_reject_discards_extract(tmp_storage, fake_embed):
    """Gate REJECT on combined path: no stored nodes (summary may start speculatively)."""
    long_text = "x" * 1600
    extract_reject = {
        **EXTRACT_RESPONSE,
        "identity": [],
        "facts": [],
        "gate_decision": {
            "novelty": 0.05,
            "biographical_relevance": 0.05,
            "emotional_arousal": 0.0,
            "reason": "low value",
        },
    }
    llm = FakeLLMClient(responses={"extract": extract_reject, "text": "不应被调用"})
    factory = ComponentFactory(
        settings=Settings(
            mode="system1",
            storage_dir=tmp_storage,
            gate_enabled=True,
            combined_gate_extract=True,
            summarizer_enabled=True,
            summarizer_min_content_tokens=600,
        ),
        embed=fake_embed,
        llm=llm,
    )
    agent = MemAgent(factory=factory)
    raw = _raw(fake_embed, long_text)
    factory.vector.upsert([raw])

    stored_ids, gate_result, _ = await agent.run(
        raw_node=raw,
        content=long_text,
        embedding=raw.embedding,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req-reject",
        memory_at=None,
    )

    assert gate_result.passed is False
    assert stored_ids == []


async def test_combined_pass_summarizer_overlaps_extract(tmp_storage, fake_embed):
    """Long PASS turn: summarizer starts before extract returns (speculative overlap)."""
    long_text = "x" * 1600
    order: list[str] = []

    llm = FakeLLMClient(responses={"extract": EXTRACT_RESPONSE, "text": "摘要"})
    factory = ComponentFactory(
        settings=Settings(
            mode="system1",
            storage_dir=tmp_storage,
            gate_enabled=True,
            combined_gate_extract=True,
            summarizer_enabled=True,
            summarizer_min_content_tokens=600,
        ),
        embed=fake_embed,
        llm=llm,
    )
    agent = MemAgent(factory=factory)
    raw = _raw(fake_embed, long_text)
    factory.vector.upsert([raw])

    original_extract = agent.extractor.extract

    async def timed_extract(**kwargs):
        order.append("extract_start")
        await asyncio.sleep(0)
        result = await original_extract(**kwargs)
        order.append("extract_end")
        return result

    agent.extractor.extract = timed_extract  # type: ignore[method-assign]

    original_summarize = agent.summarizer.summarize

    async def timed_summarize(**kwargs):
        order.append("summary_start")
        result = await original_summarize(**kwargs)
        order.append("summary_end")
        return result

    agent.summarizer.summarize = timed_summarize  # type: ignore[method-assign]

    stored_ids, gate_result, _ = await agent.run(
        raw_node=raw,
        content=long_text,
        embedding=raw.embedding,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req-overlap",
        memory_at=None,
    )

    assert gate_result.passed is True
    assert len(stored_ids) >= 2
    assert "summary_start" in order and "extract_start" in order
    assert order.index("summary_start") < order.index("extract_end")


async def test_summary_failure_still_persists_l2_l4(tmp_storage, fake_embed):
    """Speculative summary failure must not abort add after fast_write."""
    long_text = "用户" + "聊了很多关于旅行和美食的事情。" * 110
    llm = FakeLLMClient(responses={"extract": EXTRACT_RESPONSE})
    factory = ComponentFactory(
        settings=Settings(
            mode="system1",
            storage_dir=tmp_storage,
            gate_enabled=False,
            summarizer_enabled=True,
            summarizer_min_content_tokens=600,
        ),
        embed=fake_embed,
        llm=llm,
    )
    agent = MemAgent(factory=factory)
    raw = _raw(fake_embed, long_text)
    factory.vector.upsert([raw])

    async def fail_summarize(**kwargs):
        raise TimeoutError("summary LLM timeout")

    agent.summarizer.summarize = fail_summarize  # type: ignore[method-assign]

    stored_ids, gate_result, is_ephemeral = await agent.run(
        raw_node=raw,
        content=long_text,
        embedding=raw.embedding,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req-summary-fail",
        memory_at=None,
    )

    assert is_ephemeral is False
    assert gate_result.passed is True
    assert len(stored_ids) == 2
    layers = {factory.vector.get(nid).layer for nid in stored_ids}
    assert layers == {Layer.L4_IDENTITY, Layer.L2_FACT}
    assert Layer.L3_SUMMARY not in layers


async def test_run_produces_l2_l4(tmp_storage, fake_embed):
    factory = _factory(tmp_storage, fake_embed, {"extract": EXTRACT_RESPONSE, "search_query": []})
    agent = MemAgent(factory=factory)
    raw = _raw(fake_embed, "用户喜欢喝咖啡，昨天去了北京")
    factory.vector.upsert([raw])

    stored_ids, gate_result, is_ephemeral = await agent.run(
        raw_node=raw,
        content="用户喜欢喝咖啡，昨天去了北京",
        embedding=raw.embedding,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req-1",
        memory_at=None,
    )

    assert is_ephemeral is False
    assert gate_result.passed is True
    assert len(stored_ids) == 2
    layers = {factory.vector.get(nid).layer for nid in stored_ids}
    assert layers == {Layer.L4_IDENTITY, Layer.L2_FACT}
    for nid in stored_ids:
        node = factory.vector.get(nid)
        assert node.status is MemoryStatus.ACTIVE
        assert node.is_latest is True


async def test_fast_write_batches_l0_with_l2_l4(tmp_storage, fake_embed):
    response = {
        **EXTRACT_RESPONSE,
        "basic_info": {"name": "张三"},
    }
    factory = _factory(tmp_storage, fake_embed, {"extract": response, "search_query": []})
    agent = MemAgent(factory=factory)
    raw = _raw(fake_embed, "我叫张三，喜欢咖啡")
    factory.vector.upsert([raw])

    stored_ids, _, _ = await agent.run(
        raw_node=raw,
        content="我叫张三，喜欢咖啡",
        embedding=raw.embedding,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id="req-l0",
        memory_at=None,
    )

    assert len(stored_ids) == 3
    layers = {factory.vector.get(nid).layer for nid in stored_ids}
    assert layers == {Layer.L4_IDENTITY, Layer.L2_FACT, Layer.L0_BASIC_INFO}


async def test_run_long_content_adds_l3(tmp_storage, fake_embed):
    long_text = "用户" + "聊了很多关于旅行和美食的事情。" * 110
    assert len(long_text) >= 1500
    factory = _factory(
        tmp_storage,
        fake_embed,
        {"extract": EXTRACT_RESPONSE, "search_query": [], "text": "用户喜欢旅行和美食。"},
    )
    agent = MemAgent(factory=factory)
    raw = _raw(fake_embed, long_text)

    stored_ids, _, _ = await agent.run(
        raw_node=raw,
        content=long_text,
        embedding=raw.embedding,
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
