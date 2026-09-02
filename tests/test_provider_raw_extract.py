"""Provider writes L1 every turn; extract on window / idle / session end."""
import asyncio

from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.integrations._base import MemoryBackend, _SyncMemoryProvider
from dual_mem.types import Layer, MemoryStatus

from conftest import FakeLLMClient


def _provider(client: MemoryClient, *, window: int = 5, idle: float = 0.0) -> _SyncMemoryProvider:
    class Runner:
        def run(self, coro):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            box: dict = {}

            def _go() -> None:
                box["out"] = asyncio.run(coro)

            t = __import__("threading").Thread(target=_go)
            t.start()
            t.join()
            return box["out"]

    provider = object.__new__(_SyncMemoryProvider)
    provider.name = "dual-mem"
    provider._backend = MemoryBackend(client=client)
    provider._runner = Runner()
    provider._user_id = "u_p"
    provider._agent_id = "agent"
    provider._session_id = "s1"
    provider._write_turn_window = window
    provider._idle_timeout_sec = idle
    provider._idle_timers = {}
    provider._l1_ids = {}
    provider._turn_buffer = {}
    provider._buffer_lock = __import__("threading").Lock()
    provider._executor = None
    provider._inflight = []
    return provider


async def test_first_turn_is_searchable_as_l1(tmp_storage, fake_embed):
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(),
    )
    provider = _provider(client, window=5)
    provider.sync_turn("我超爱美式咖啡", "记下了")
    conv = await client.search_conversation(
        query="美式咖啡", app_ids=[client.settings.default_app_id], user_id="u_p", min_score=0.0,
    )
    assert any("美式咖啡" in (m.content or "") for m in conv.memories.normal)
    facts = [
        n for n in client.factory.vector.get_many(
            {"$and": [{"app_id": client.settings.default_app_id}, {"user_id": "u_p"}]},
            limit=20,
        )
        if n.layer is Layer.L2_FACT
    ]
    assert facts == []
    await client.aclose()


async def test_window_flush_extracts_and_shadows_turn_l1(tmp_storage, fake_embed):
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
    provider = _provider(client, window=2, idle=0)
    provider.sync_turn("我超爱美式", "好")
    provider.sync_turn("每天早上都喝", "好")
    facts = [
        n for n in client.factory.vector.get_many(
            {"$and": [
                {"app_id": client.settings.default_app_id},
                {"user_id": "u_p"},
                {"status": "ACTIVE"},
            ]},
            limit=20,
        )
        if n.layer is Layer.L2_FACT
    ]
    assert facts
    l1s = [
        n for n in client.factory.vector.get_many(
            {"$and": [{"app_id": client.settings.default_app_id}, {"user_id": "u_p"}]},
            limit=20,
        )
        if n.layer is Layer.L1_RAW
    ]
    assert l1s
    assert all(n.status is MemoryStatus.SHADOW for n in l1s)
    conv = await client.search_conversation(
        query="美式", app_ids=[client.settings.default_app_id], user_id="u_p", min_score=0.0,
    )
    assert any("美式" in (m.content or "") for m in conv.memories.normal)
    await client.aclose()


async def test_idle_flush_extracts(tmp_storage, fake_embed):
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses={
            "extract": {
                "is_ephemeral": False,
                "identity": [],
                "facts": [{"content": "用户想喝茶", "tags": ["饮品"]}],
                "intentions": [],
                "basic_info": {},
            }
        }),
    )
    provider = _provider(client, window=5)
    provider.sync_turn("我今天想喝茶", "好")
    provider._idle_flush(provider._session_id)
    facts = [
        n for n in client.factory.vector.get_many(
            {"$and": [
                {"app_id": client.settings.default_app_id},
                {"user_id": "u_p"},
                {"status": "ACTIVE"},
            ]},
            limit=20,
        )
        if n.layer is Layer.L2_FACT
    ]
    assert facts
    await client.aclose()


async def test_session_end_flushes_partial_window(tmp_storage, fake_embed):
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses={
            "extract": {
                "is_ephemeral": False,
                "identity": [],
                "facts": [{"content": "用户喜欢美式", "tags": ["饮品"]}],
                "intentions": [],
                "basic_info": {},
            }
        }),
    )
    provider = _provider(client, window=5)
    provider.sync_turn("我超爱美式咖啡", "记下了")
    provider.on_session_end([])
    facts = [
        n for n in client.factory.vector.get_many(
            {"$and": [
                {"app_id": client.settings.default_app_id},
                {"user_id": "u_p"},
                {"status": "ACTIVE"},
            ]},
            limit=20,
        )
        if n.layer is Layer.L2_FACT
    ]
    assert facts
    await client.aclose()
