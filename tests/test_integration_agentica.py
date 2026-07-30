# -*- coding: utf-8 -*-
"""agentica 集成的单元测试（全 mock，不连真实 LLM / Embed / 网络）。

覆盖：
- ``DualMemMemory``（MemoryBackend 扩展）：add_messages / retrieve
- ``get_backend("agentica", ...)`` 工厂
- ``DualMemWorkspace``（Agent 的 workspace 适配器）：语义召回 / 去重 / 写入 / 转发
- ``get_agentica_memory_backend`` 工厂
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from dual_mem.integrations import get_backend
from dual_mem.integrations.agentica import (
    DualMemMemory,
    DualMemWorkspace,
    get_agentica_memory_backend,
)
from dual_mem.sdk_models import MemoryItem, SearchMemories, SearchResult


def _make_client() -> SimpleNamespace:
    """一个内存版假 MemoryClient：记录写入、按 query 返回固定记忆。"""

    class FakeClient:
        def __init__(self) -> None:
            self.adds: list[dict] = []
            self.searches: list[dict] = []

        async def add(self, **kw: object) -> SimpleNamespace:
            self.adds.append(kw)
            return SimpleNamespace(memory_id="mem-001")

        async def search(self, **kw: object) -> SearchResult:
            self.searches.append(kw)
            items = [
                MemoryItem(
                    memory_id="mem-001",
                    content="用户喜欢燕麦拿铁",
                    category="L2_FACT",
                    score=0.9,
                ),
                MemoryItem(
                    memory_id="mem-002",
                    content="用户住在北京",
                    category="L0_BASIC_INFO",
                    score=0.8,
                ),
            ]
            return SearchResult(
                success=True,
                request_id="req-1",
                memories=SearchMemories(normal=items),
                processing_time_ms=1.0,
            )

    return FakeClient()


async def test_get_backend_returns_dualmem_memory() -> None:
    mem = get_backend("agentica", user_id="u1", client=_make_client())
    assert isinstance(mem, DualMemMemory)
    assert mem._user_id == "u1"


async def test_dualmem_add_messages_routes_to_client() -> None:
    client = _make_client()
    mem = DualMemMemory(user_id="u1", client=client)
    msgs = [
        {"role": "user", "content": "我喜欢燕麦拿铁"},
        {"role": "assistant", "content": "好的"},
    ]
    res = await mem.add_messages(msgs)
    assert res.memory_id == "mem-001"
    assert len(client.adds) == 1
    assert client.adds[0]["messages"] == msgs
    assert client.adds[0]["user_id"] == "u1"


async def test_dualmem_retrieve_renders_prompt_block() -> None:
    mem = DualMemMemory(user_id="u1", client=_make_client())
    ctx = await mem.retrieve("用户喜欢什么咖啡?", user_id="u1")
    assert ctx.startswith("<relevant-memories>")
    assert "用户喜欢燕麦拿铁" in ctx
    assert "用户住在北京" in ctx


async def test_dualmem_user_id_required() -> None:
    mem = DualMemMemory(client=_make_client())
    with pytest.raises(ValueError):
        await mem.retrieve("query")


@pytest.mark.skipif(
    __import__("shutil").which("agentica") is not None and False,
    reason="never",
)
async def test_workspace_recall_and_dedup() -> None:
    pytest.importorskip("agentica")
    mem = DualMemMemory(user_id="u1", client=_make_client())
    ws = mem.as_workspace(workspace_dir="/tmp/_dmws_test")

    out = await ws.get_relevant_memories("用户喜欢什么咖啡?", limit=5)
    assert out.startswith("Relevant memories for")
    assert "## Memory" in out
    assert "用户喜欢燕麦拿铁" in out
    assert "用户住在北京" in out

    # 去重：已出现的记忆不应再次返回
    out2 = await ws.get_relevant_memories(
        "用户喜欢什么咖啡?",
        limit=5,
        already_surfaced=[("t", "用户喜欢燕麦拿铁")],
    )
    assert "用户喜欢燕麦拿铁" not in out2
    assert "用户住在北京" in out2


async def test_workspace_save_memory_routes_to_client() -> None:
    pytest.importorskip("agentica")
    client = _make_client()
    mem = DualMemMemory(user_id="u1", client=client)
    ws = mem.as_workspace(workspace_dir="/tmp/_dmws_test")

    res = await ws.save_memory("用户不吃香菜")
    assert res.memory_id == "mem-001"
    assert client.adds[-1]["content"] == "用户不吃香菜"
    assert client.adds[-1]["user_id"] == "u1"


async def test_workspace_forwards_file_methods() -> None:
    pytest.importorskip("agentica")
    mem = DualMemMemory(user_id="u1", client=_make_client())
    ws = mem.as_workspace(workspace_dir="/tmp/_dmws_test")
    # 文件型上下文方法转发给原生 Workspace
    assert ws.path is not None
    assert isinstance(ws.exists(), bool)
    # Workspace.get_context_prompt 是异步的（Agent 内部 await），转发后仍是 coroutine
    assert isinstance(await ws.get_context_prompt(), str)


def test_workspace_forwards_all_native_public_methods() -> None:
    from agentica.workspace import Workspace

    native_public = {
        name
        for name, value in Workspace.__dict__.items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.iscoroutinefunction(value))
    }
    adapter_public = {
        name
        for name, value in DualMemWorkspace.__dict__.items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.iscoroutinefunction(value))
    }

    assert native_public <= adapter_public
    for name in native_public:
        adapter_signature = inspect.signature(getattr(DualMemWorkspace, name))
        native_signature = inspect.signature(getattr(Workspace, name))
        assert list(adapter_signature.parameters) == list(native_signature.parameters)
        for parameter_name, native_parameter in native_signature.parameters.items():
            adapter_parameter = adapter_signature.parameters[parameter_name]
            assert adapter_parameter.kind == native_parameter.kind
            assert adapter_parameter.default == native_parameter.default


async def test_get_agentica_memory_backend_returns_workspace() -> None:
    pytest.importorskip("agentica")
    ws = get_agentica_memory_backend(
        client=_make_client(),
        user_id="u1",
        workspace_dir="/tmp/_dmws_test",
    )
    assert isinstance(ws, DualMemWorkspace)
    assert ws.user_id == "u1"
    out = await ws.get_relevant_memories("用户喜欢什么咖啡?", limit=3)
    assert "用户喜欢燕麦拿铁" in out
