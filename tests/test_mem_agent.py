import asyncio

from dual_mem.agent.mem_agent import MemAgent
from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryStatus

from conftest import FakeLLMClient


EXTRACT_RESPONSE = {
    "is_ephemeral": False,
    "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
    "identity": [{"content": "用户喜欢喝咖啡", "speculate": None, "tags": ["food"]}],
    "facts": [{"content": "用户昨天去了北京", "speculate": None, "tags": ["travel"]}],
    "intentions": [],
    "basic_info": {},
}


def _factory(tmp_storage, fake_embed, responses, **settings_kwargs):
    return ComponentFactory(
        settings=Settings(
            mode="system1",
            storage_dir=tmp_storage,
            **settings_kwargs,
        ),
        embed=fake_embed,
        llm=FakeLLMClient(responses=responses),
    )


async def _run(agent: MemAgent, content: str, request_id: str = "req"):
    return await agent.run(
        content=content,
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
        request_id=request_id,
        memory_at=None,
    )


async def test_extract_commit_uses_one_llm_call_and_writes_l2_l4(tmp_storage, fake_embed):
    factory = _factory(tmp_storage, fake_embed, {"extract": EXTRACT_RESPONSE})
    agent = MemAgent(factory=factory)

    stored_ids, commit_result, is_ephemeral = await _run(
        agent,
        "用户喜欢喝咖啡，昨天去了北京",
    )

    assert is_ephemeral is False
    assert commit_result.passed is True
    assert commit_result.reason == "extractor produced persistable memory"
    assert len(stored_ids) == 2
    assert [call["type"] for call in factory.llm.calls] == ["chat_json"]
    layers = {factory.vector.get(node_id).layer for node_id in stored_ids}
    assert layers == {Layer.L4_IDENTITY, Layer.L2_FACT}
    for node_id in stored_ids:
        node = factory.vector.get(node_id)
        assert node.status is MemoryStatus.ACTIVE
        assert node.is_latest is True


async def test_ephemeral_extract_rejects_commit(tmp_storage, fake_embed):
    response = {**EXTRACT_RESPONSE, "is_ephemeral": True}
    factory = _factory(tmp_storage, fake_embed, {"extract": response})

    stored_ids, commit_result, is_ephemeral = await _run(
        MemAgent(factory=factory),
        "嗯嗯好的",
    )

    assert stored_ids == []
    assert commit_result.passed is False
    assert commit_result.reason == "extractor marked content ephemeral"
    assert is_ephemeral is True


async def test_empty_extract_rejects_commit(tmp_storage, fake_embed):
    response = {
        **EXTRACT_RESPONSE,
        "identity": [],
        "facts": [],
        "intentions": [],
        "basic_info": {},
    }
    factory = _factory(tmp_storage, fake_embed, {"extract": response})

    stored_ids, commit_result, is_ephemeral = await _run(
        MemAgent(factory=factory),
        "没有可沉淀的信息",
    )

    assert stored_ids == []
    assert commit_result.passed is False
    assert commit_result.reason == "extractor produced no persistable memory"
    assert is_ephemeral is False


async def test_blank_structured_items_reject_commit(tmp_storage, fake_embed):
    response = {
        **EXTRACT_RESPONSE,
        "identity": [{"content": "   "}],
        "facts": [{"content": ""}],
        "intentions": [{"content": "\t"}],
        "basic_info": {"nickname": "not a supported L0 field"},
    }
    factory = _factory(tmp_storage, fake_embed, {"extract": response})

    stored_ids, commit_result, is_ephemeral = await _run(
        MemAgent(factory=factory),
        "Extractor 返回了无效空白项",
    )

    assert stored_ids == []
    assert commit_result.passed is False
    assert commit_result.reason == "extractor produced no persistable memory"
    assert is_ephemeral is False


async def test_passed_extract_summarizer_overlaps(tmp_storage, fake_embed):
    long_text = "x" * 1600
    order: list[str] = []
    factory = _factory(
        tmp_storage,
        fake_embed,
        {"extract": EXTRACT_RESPONSE, "text": "摘要"},
        summarizer_enabled=True,
        summarizer_min_content_tokens=600,
    )
    agent = MemAgent(factory=factory)
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

    stored_ids, commit_result, _ = await _run(agent, long_text, "req-overlap")

    assert commit_result.passed is True
    assert len(stored_ids) == 3
    assert order.index("summary_start") < order.index("extract_end")


async def test_summary_failure_still_persists_l2_l4(tmp_storage, fake_embed):
    long_text = "用户" + "聊了很多关于旅行和美食的事情。" * 110
    factory = _factory(
        tmp_storage,
        fake_embed,
        {"extract": EXTRACT_RESPONSE},
        summarizer_enabled=True,
        summarizer_min_content_tokens=600,
    )
    agent = MemAgent(factory=factory)

    async def fail_summarize(**kwargs):
        raise TimeoutError("summary LLM timeout")

    agent.summarizer.summarize = fail_summarize  # type: ignore[method-assign]

    stored_ids, commit_result, is_ephemeral = await _run(
        agent,
        long_text,
        "req-summary-fail",
    )

    assert is_ephemeral is False
    assert commit_result.passed is True
    assert len(stored_ids) == 2
    layers = {factory.vector.get(node_id).layer for node_id in stored_ids}
    assert layers == {Layer.L4_IDENTITY, Layer.L2_FACT}


async def test_fast_write_batches_l0_with_l2_l4(tmp_storage, fake_embed):
    response = {**EXTRACT_RESPONSE, "basic_info": {"name": "张三"}}
    factory = _factory(tmp_storage, fake_embed, {"extract": response})

    stored_ids, commit_result, _ = await _run(
        MemAgent(factory=factory),
        "我叫张三，喜欢咖啡",
        "req-l0",
    )

    assert commit_result.passed is True
    assert len(stored_ids) == 3
    layers = {factory.vector.get(node_id).layer for node_id in stored_ids}
    assert layers == {Layer.L4_IDENTITY, Layer.L2_FACT, Layer.L0_BASIC_INFO}
