"""L1_RAW evidence search is opt-in and never mixed into default QA search."""
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


async def test_search_conversation_finds_shadowed_l1(tmp_storage, fake_embed):
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
    written = await client.add(
        content="我超爱美式咖啡，每天早上都喝",
        app_id="app",
        user_id="u_raw",
    )
    raw = client.factory.vector.get(written.memory_id)
    assert raw is not None
    assert raw.layer is Layer.L1_RAW
    assert raw.status is MemoryStatus.SHADOW

    qa = await client.search(query="美式咖啡", app_ids=["app"], user_id="u_raw", min_score=0.0)
    assert all(item.memory_id != written.memory_id for item in qa.memories.flatten())

    conv = await client.search_conversation(
        query="美式咖啡", app_ids=["app"], user_id="u_raw", min_score=0.0,
    )
    ids = [item.memory_id for item in conv.memories.normal]
    assert written.memory_id in ids
    assert all(item.category == "raw" for item in conv.memories.normal)
    await client.aclose()


async def test_search_conversation_keeps_active_l1_when_no_derived(tmp_storage, fake_embed):
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses={
            "extract": {
                "is_ephemeral": True,
                "identity": [],
                "facts": [],
                "intentions": [],
                "basic_info": {},
            }
        }),
    )
    written = await client.add(
        content="随便聊聊天气",
        app_id="app",
        user_id="u_ephemeral",
    )
    raw = client.factory.vector.get(written.memory_id)
    assert raw.status is MemoryStatus.ACTIVE
    conv = await client.search_conversation(
        query="天气", app_ids=["app"], user_id="u_ephemeral", min_score=0.0,
    )
    assert written.memory_id in [item.memory_id for item in conv.memories.normal]
    await client.aclose()
