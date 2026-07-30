"""Pipeline trace for Extract commit decisions and hybrid read stages."""
from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


_EXTRACT = {
    "is_ephemeral": False,
    "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
    "identity": [{"content": "用户喜欢喝咖啡", "speculate": None, "tags": ["food"]}],
    "facts": [],
    "intentions": [],
    "basic_info": {},
}


async def test_write_logs_extract_commit_decision(tmp_storage, fake_embed):
    """add() writes the Extract payload and its commit decision in one stage."""
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(
        settings=settings, embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": _EXTRACT, "search_query": []}),
    )
    result = await client.add(content="用户最近开始喝咖啡了，每天都要来一杯", app_id="app", user_id="u")
    logs = client.factory.cache.list_pipeline_logs(result.request_id)
    stages = {entry["stage"] for entry in logs}
    assert "EXTRACT" in stages
    assert "GATE" not in stages
    extract_log = next(entry for entry in logs if entry["stage"] == "EXTRACT")
    assert extract_log["payload"]["commit_passed"] is True
    assert extract_log["payload"]["commit_reason"] == "extractor produced persistable memory"

    await client.aclose()


async def test_search_logs_qu_anchor_expand_fusion(tmp_storage, fake_embed):
    """search() with explicit request_id → READ_QU + READ_HYBRID stages logged."""
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(
        settings=settings, embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": _EXTRACT}),
    )
    # Seed one node so the read path actually has something to recall.
    node = MemoryNode(
        content="用户喜欢喝咖啡", layer=Layer.L4_IDENTITY,
        app_id="app", user_id="u", status=MemoryStatus.ACTIVE,
    )
    node.embedding = fake_embed.embed_sync(node.content)
    client.factory.vector.upsert([node])

    result = await client.search(
        query="咖啡", app_ids=["app"], user_id="u", request_id="rid-test-1",
    )
    assert result.success
    logs = client.factory.cache.list_pipeline_logs("rid-test-1")
    stages = {entry["stage"] for entry in logs}
    assert "READ_QU" in stages
    assert "READ_HYBRID" in stages

    await client.aclose()
