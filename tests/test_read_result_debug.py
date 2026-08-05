"""R7: ReadResult 暴露 — debug=True 走 reader.search(collect_trace=True) 返回完整 trace。"""
from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


async def test_search_debug_returns_read_result(tmp_storage, fake_embed):
    """client.search(debug=True) → SearchResult.read_result 非空，含 final_count。"""
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={}))

    n = MemoryNode(
        content="用户喜欢喝咖啡", layer=Layer.L4_IDENTITY,
        app_id="app", user_id="u", status=MemoryStatus.ACTIVE,
    )
    n.embedding = fake_embed.embed_sync(n.content)
    client.factory.vector.upsert([n])

    result = await client.search(query="咖啡", app_ids=["app"], user_id="u", debug=True)
    assert result.read_result is not None
    rr = result.read_result
    assert rr.final_count >= 1
    assert rr.elapsed_ms >= 0.0
    # to_dict 应该包含 read_result
    d = result.to_dict()
    assert "read_result" in d
    assert d["read_result"]["final_count"] >= 1

    await client.aclose()


async def test_search_default_no_read_result(tmp_storage, fake_embed):
    """默认 debug=False → SearchResult.read_result 为 None，to_dict 不含 read_result key。"""
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={}))
    result = await client.search(query="x", app_ids=["app"], user_id="u")
    assert result.read_result is None
    assert "read_result" not in result.to_dict()
    await client.aclose()
