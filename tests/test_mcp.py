import json

import pytest

from dual_mem import MemoryClient
from dual_mem.api.operations import MemoryOperations
from dual_mem.config import Settings
from dual_mem.mcp.server import build_mcp


@pytest.fixture
def mcp(tmp_storage, fake_embed, fake_llm):
    settings = Settings(storage_dir=tmp_storage, mode="system1", auth_disabled=True)
    client = MemoryClient(settings=settings, embed=fake_embed, llm=fake_llm)
    ops = MemoryOperations(client)
    return build_mcp(ops=ops)


def _payload(result):
    if len(result) > 1 and isinstance(result[1], dict) and "result" in result[1]:
        return result[1]["result"]
    content = result[0]
    if isinstance(content, list):
        text = content[0].text
    else:
        text = content.text
    parsed = json.loads(text)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    return parsed


async def test_tools_registered(mcp):
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "memory_add",
        "memory_search",
        "conversation_search",
        "memory_list",
        "memory_get",
        "memory_update",
        "memory_delete",
        "memory_delete_scope",
        "memory_list_scopes",
        "memory_digest",
    }


async def test_add_search_get_update_delete(mcp):
    added = await mcp.call_tool(
        "memory_add", {"content": "用户喜欢喝咖啡", "user_id": "u"}
    )
    body = _payload(added)
    assert body["success"] is True
    memory_id = body["memory_id"]

    got = await mcp.call_tool("memory_get", {"memory_id": memory_id})
    assert _payload(got)["memory_id"] == memory_id

    updated = await mcp.call_tool(
        "memory_update", {"memory_id": memory_id, "content": "用户喜欢喝茶"}
    )
    assert _payload(updated)["success"] is True

    listed = await mcp.call_tool(
        "memory_list", {"user_id": "u", "limit": 10}
    )
    assert isinstance(_payload(listed), list)

    searched = await mcp.call_tool(
        "memory_search",
        {"query": "饮品", "user_id": "u", "min_score": 0.0},
    )
    memories = _payload(searched)["memories"]
    assert set(memories.keys()) == {"profile", "proactive", "normal"}

    deleted = await mcp.call_tool("memory_delete", {"memory_id": memory_id})
    assert _payload(deleted)["success"] is True


async def test_list_scopes_and_delete_scope(mcp):
    await mcp.call_tool(
        "memory_add", {"content": "scope test", "app_id": "app2", "user_id": "u2"}
    )
    scopes = await mcp.call_tool("memory_list_scopes", {"app_id": "app2"})
    rows = _payload(scopes)
    assert any(r["user_id"] == "u2" for r in rows)

    rejected = await mcp.call_tool(
        "memory_delete_scope", {"app_id": "app2", "user_id": "u2", "confirm": False}
    )
    assert _payload(rejected)["success"] is False

    ok = await mcp.call_tool(
        "memory_delete_scope", {"app_id": "app2", "user_id": "u2", "confirm": True}
    )
    assert _payload(ok)["success"] is True
    assert _payload(ok)["deleted"] >= 1
