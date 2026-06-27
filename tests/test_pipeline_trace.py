"""P2-1: pipeline_logs trace — write GATE/EXTRACT, search READ_QU/READ_ANCHOR/.. stages."""
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


async def test_write_logs_gate_and_extract(tmp_storage, fake_embed):
    """add() 写完后，pipeline_logs 中应该有 GATE 和 EXTRACT 两个 stage 条目。"""
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(
        settings=settings, embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": _EXTRACT, "search_query": []}),
    )
    result = await client.add(content="用户最近开始喝咖啡了，每天都要来一杯", app_id="app", user_id="u")
    logs = client.factory.cache.list_pipeline_logs(result.request_id)
    stages = {entry["stage"] for entry in logs}
    assert "GATE" in stages
    assert "EXTRACT" in stages
    # GATE payload contains the score
    gate_log = next(e for e in logs if e["stage"] == "GATE")
    assert "score" in gate_log["payload"]
    assert "passed" in gate_log["payload"]

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
