import json

import pytest

from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.mcp.server import build_mcp


@pytest.fixture
def mcp(tmp_storage, fake_embed, fake_llm):
    settings = Settings(storage_dir=tmp_storage, mode="system1", auth_disabled=True)
    client = MemoryClient(settings=settings, embed=fake_embed, llm=fake_llm)
    return build_mcp(client=client)


def _payload(result):
    return json.loads(result[0].text)


async def test_tools_registered(mcp):
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "memory_add",
        "memory_search",
        "memory_get",
        "memory_list",
        "memory_delete",
    }


async def test_add_then_search(mcp):
    added = await mcp.call_tool(
        "memory_add", {"content": "用户喜欢喝咖啡", "app_id": "app", "user_id": "u"}
    )
    body = _payload(added)
    assert body["success"] is True
    memory_id = body["memory_id"]

    searched = await mcp.call_tool(
        "memory_search",
        {"query": "用户喜欢喝咖啡", "app_ids": ["app"], "user_id": "u", "min_score": 0.4},
    )
    memories = _payload(searched)["memories"]
    assert set(memories.keys()) == {"profile", "proactive", "normal"}
    assert any(m["memory_id"] == memory_id for m in memories["normal"])
