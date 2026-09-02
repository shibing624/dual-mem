"""add_raw writes L1 only; distill extracts without a second L1."""
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryStatus

from conftest import FakeLLMClient


async def test_add_raw_does_not_call_extract(tmp_storage, fake_embed):
    llm = FakeLLMClient()
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=llm,
    )
    result = await client.add_raw(
        content="我超爱美式咖啡",
        app_id="app",
        user_id="u_raw",
    )
    assert result.success
    assert result.memory_id
    assert result.extracted_count == 0
    raw = client.factory.vector.get(result.memory_id)
    assert raw is not None
    assert raw.layer is Layer.L1_RAW
    assert raw.status is MemoryStatus.ACTIVE
    assert not any(
        c["type"] == "chat_json" and ("记忆分析专家" in c["system"] or "memory analyst" in c["system"])
        for c in llm.calls
    )
    await client.aclose()


async def test_distill_extracts_and_shadows_source_l1(tmp_storage, fake_embed):
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses={
            "extract": {
                "is_ephemeral": False,
                "identity": [],
                "facts": [{"content": "用户喜欢喝美式咖啡", "tags": ["饮品"]}],
                "intentions": [],
                "basic_info": {},
            }
        }),
    )
    first = await client.add_raw(content="我超爱美式", app_id="app", user_id="u_d")
    second = await client.add_raw(content="每天早上都喝", app_id="app", user_id="u_d")
    distilled = await client.distill(
        messages=[
            {"role": "user", "content": "我超爱美式"},
            {"role": "assistant", "content": "好的"},
            {"role": "user", "content": "每天早上都喝"},
        ],
        app_id="app",
        user_id="u_d",
        source_node_ids=[first.memory_id, second.memory_id],
    )
    assert distilled.extracted_count >= 1
    assert client.factory.vector.get(first.memory_id).status is MemoryStatus.SHADOW
    assert client.factory.vector.get(second.memory_id).status is MemoryStatus.SHADOW
    l1s = [
        n for n in client.factory.vector.get_many(
            {"$and": [{"app_id": "app"}, {"user_id": "u_d"}]}, limit=20
        )
        if n.layer is Layer.L1_RAW
    ]
    assert len(l1s) == 2
    facts = [
        n for n in client.factory.vector.get_many(
            {"$and": [{"app_id": "app"}, {"user_id": "u_d"}, {"status": "ACTIVE"}]},
            limit=20,
        )
        if n.layer is Layer.L2_FACT
    ]
    assert facts
    assert (facts[0].custom or {}).get("source_node_id") == first.memory_id
    await client.aclose()


async def test_get_returns_shadowed_l1_source(tmp_storage, fake_embed):
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses={
            "extract": {
                "is_ephemeral": False,
                "identity": [],
                "facts": [{"content": "用户喜欢茶", "tags": ["饮品"]}],
                "intentions": [],
                "basic_info": {},
            }
        }),
    )
    written = await client.add(content="我喜欢喝茶", app_id="app", user_id="u_src")
    raw = client.factory.vector.get(written.memory_id)
    assert raw.status is MemoryStatus.SHADOW
    got = await client.get(written.memory_id)
    assert got is not None
    assert got.content == raw.content
    facts = [
        n for n in client.factory.vector.get_many(
            {"$and": [{"app_id": "app"}, {"user_id": "u_src"}, {"status": "ACTIVE"}]},
            limit=20,
        )
        if n.layer is Layer.L2_FACT
    ]
    item = await client.get(facts[0].node_id)
    assert item.source_node_id == written.memory_id
    await client.aclose()


async def test_dual_distill_enqueues_s2_and_does_not_write_second_l1(tmp_storage, fake_embed):
    client = MemoryClient(
        settings=Settings(mode="dual", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses={
            "extract": {
                "is_ephemeral": False,
                "identity": [],
                "facts": [{"content": "用户喜欢跑步", "tags": ["运动"]}],
                "intentions": [],
                "basic_info": {},
            }
        }),
    )
    raw = await client.add_raw(content="我每天跑步", app_id="app", user_id="u_dual")
    distilled = await client.distill(
        content="我每天跑步",
        app_id="app",
        user_id="u_dual",
        source_node_ids=[raw.memory_id],
    )
    assert distilled.extracted_count >= 1
    l1s = [
        n for n in client.factory.vector.get_many(
            {"$and": [{"app_id": "app"}, {"user_id": "u_dual"}]}, limit=20
        )
        if n.layer is Layer.L1_RAW
    ]
    assert len(l1s) == 1
    assert l1s[0].status is MemoryStatus.SHADOW
    pending = client.factory.cache.conn.execute(
        "SELECT COUNT(*) FROM s2_queue WHERE status = 'pending'"
    ).fetchone()[0]
    assert pending >= 1
    await client.aclose()
