"""P1-1 子PR4 + P1-3: 默认 hybrid reader 路径回归。

These tests pin the contract that hybrid mode (default) and legacy mode both produce a
SearchMemories with profile/proactive/normal routes, and that hybrid runs the V2 anchor +
fusion + expander pipeline. They do NOT validate every dimension of fusion (those have
their own unit tests) — just the integration smoke.
"""
from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


def _seed(client, content: str, layer: Layer, *, user_id: str = "u",
          gmt: int = 1_000_000) -> str:
    n = MemoryNode(
        content=content,
        layer=layer,
        app_id="app",
        user_id=user_id,
        status=MemoryStatus.ACTIVE,
        gmt_created=gmt,
    )
    n.embedding = client.factory.embed.embed_sync(content)
    client.factory.vector.upsert([n])
    return n.node_id


async def test_hybrid_reader_default_returns_routes(tmp_storage, fake_embed):
    """默认 hybrid 模式：profile 路由召回 L4 identity，normal 路由召回 L2 fact。"""
    settings = Settings(mode="system1", storage_dir=tmp_storage, gate_enabled=False)
    assert settings.reader_mode == "hybrid"  # default
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={"extract": {
                              "is_ephemeral": False,
                              "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
                              "identity": [], "facts": [], "intentions": [], "basic_info": {}}}))

    _seed(client, "用户喜欢喝咖啡", Layer.L4_IDENTITY)
    _seed(client, "用户昨天去了北京", Layer.L2_FACT)

    result = await client.search(query="咖啡", app_ids=["app"], user_id="u")
    assert result.success
    # profile 路由命中 identity
    assert any("咖啡" in m.content for m in result.memories.profile)

    await client.aclose()


async def test_legacy_reader_mode_kept_for_baseline(tmp_storage, fake_embed):
    """legacy 模式仍然可用（BM25+RRF 基线路径）。"""
    settings = Settings(mode="system1", storage_dir=tmp_storage,
                        gate_enabled=False, reader_mode="legacy")
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={"extract": {
                              "is_ephemeral": False,
                              "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
                              "identity": [], "facts": [], "intentions": [], "basic_info": {}}}))

    _seed(client, "用户喜欢喝咖啡", Layer.L4_IDENTITY)
    result = await client.search(query="咖啡", app_ids=["app"], user_id="u")
    assert result.success
    assert any("咖啡" in m.content for m in result.memories.profile)

    await client.aclose()


async def test_hybrid_recalls_facts_too(tmp_storage, fake_embed):
    """hybrid 模式 normal 路由也要能命中 L2 fact。"""
    settings = Settings(mode="system1", storage_dir=tmp_storage, gate_enabled=False)
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={"extract": {
                              "is_ephemeral": False,
                              "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
                              "identity": [], "facts": [], "intentions": [], "basic_info": {}}}))

    _seed(client, "用户去年加入了阿里巴巴", Layer.L2_FACT)
    result = await client.search(query="阿里巴巴", app_ids=["app"], user_id="u",
                                 min_score=0.0)
    # 至少有一处 normal 命中
    assert any("阿里" in m.content for m in result.memories.normal)

    await client.aclose()
