import json
import sys

import pytest

pytest.importorskip(
    "mcp.server.mcpserver",
    reason="dual-mem's MCP server requires mcp 2.x (install the 'mcp' extra)",
)

from dual_mem import MemoryClient  # noqa: E402
from dual_mem.operations import MemoryOperations  # noqa: E402
from dual_mem.config import Settings  # noqa: E402
from dual_mem.mcp.server import build_mcp, run_server  # noqa: E402


@pytest.fixture
def mcp(tmp_storage, fake_embed, fake_llm):
    settings = Settings(storage_dir=tmp_storage, mode="system1", auth_disabled=True)
    client = MemoryClient(settings=settings, embed=fake_embed, llm=fake_llm)
    ops = MemoryOperations(client)
    return build_mcp(ops=ops)


def _payload(result):
    """The tool's return value, from an mcp 2.x ``CallToolResult``.

    Non-dict returns (lists, ``None``) are wrapped as ``structured_content={"result": ...}``;
    dict returns come back only as one JSON text block.
    """
    assert result.is_error is False, result.content
    if result.structured_content is not None:
        return result.structured_content["result"]
    (block,) = result.content
    return json.loads(block.text)


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
    schema = next(t for t in tools if t.name == "memory_search").input_schema
    assert schema["required"] == ["query", "user_id"]
    assert schema["properties"]["procedure_limit"]["default"] == 0


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
    updated_body = _payload(updated)
    assert updated_body["success"] is True
    new_id = updated_body["memory_id"]
    assert new_id != memory_id

    listed = await mcp.call_tool(
        "memory_list", {"user_id": "u", "limit": 10}
    )
    assert isinstance(_payload(listed), list)

    searched = await mcp.call_tool(
        "memory_search",
        {"query": "饮品", "user_id": "u", "min_score": 0.0},
    )
    memories = _payload(searched)["memories"]
    assert set(memories.keys()) == {
        "profile", "proactive", "normal", "raw", "schema", "catalog", "procedures"
    }

    # The superseded version is not current: archiving it is refused (REST: 409).
    stale = await mcp.call_tool("memory_delete", {"memory_id": memory_id})
    assert _payload(stale)["success"] is False
    deleted = await mcp.call_tool("memory_delete", {"memory_id": new_id})
    assert _payload(deleted)["success"] is True


async def test_get_missing_returns_null(mcp):
    missing = await mcp.call_tool("memory_get", {"memory_id": "nope"})
    assert _payload(missing) is None


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


def test_run_server_rejects_unknown_transport():
    with pytest.raises(ValueError, match="sse"):
        run_server(transport="sse")


async def test_stdio_round_trip_with_a_real_client(tmp_path):
    """Spawn ``python -m dual_mem.mcp.server`` and speak MCP to it over stdio.

    Pins the wiring the in-process tests cannot see: the stdio transport, the
    initialize handshake and the stdout discipline (one stray print breaks JSON-RPC).
    No model is called: ``list_tools`` and an unauthenticated scope listing on an
    empty store never touch the LLM or the embedder.
    """
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "dual_mem.mcp.server"],
        env={
            "DUAL_MEM_STORAGE_DIR": str(tmp_path / "store"),
            "DUAL_MEM_AUTH_DISABLED": "true",
            "DUAL_MEM_LLM_API_KEY": "unused",
            "DUAL_MEM_EMBED_API_KEY": "unused",
            "PATH": "/usr/bin:/bin",
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert "memory_search" in names and "memory_digest" in names
            scopes = await session.call_tool("memory_list_scopes", {})
            assert scopes.is_error is False
            assert scopes.structured_content == {"result": []}
