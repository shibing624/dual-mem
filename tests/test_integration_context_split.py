"""Query-independent profile_block, prefetch facts-only, search tool limit."""
import asyncio

from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.integrations._base import (
    MEMORY_TOOLS_GUIDE,
    MemoryBackend,
    RenderedMemoryContext,
    _SyncMemoryProvider,
    format_profile_block,
    format_topic_catalog,
)
from dual_mem.sdk_models import MemoryItem, SearchMemories, SearchResult
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


def test_format_profile_block_is_stable_across_item_order():
    a = MemoryItem(memory_id="p1", content="用户叫张三", category="profile", score=0.1)
    b = MemoryItem(memory_id="p0", content="用户住在上海", category="profile", score=0.9)
    first = format_profile_block([a, b])
    second = format_profile_block([b, a])
    assert first == second
    assert first.startswith("<user-profile>")
    assert "张三" in first
    assert "上海" in first


def test_format_profile_block_empty():
    assert format_profile_block([]) == ""


def test_format_topic_catalog_is_sorted_and_stable():
    first = format_topic_catalog(["工作", "居住"], ["支付场景", "出差"])
    second = format_topic_catalog(["居住", "工作"], ["出差", "支付场景"])
    assert first == second
    assert first.startswith("<topic-catalog>")
    assert first.index("居住") < first.index("工作")
    assert "支付场景" in first


def test_format_topic_catalog_empty():
    assert format_topic_catalog([], []) == ""


def _provider_with_backend(backend, *, runner=None) -> _SyncMemoryProvider:
    class Runner:
        def run(self, coro):
            return asyncio.run(coro)

    provider = object.__new__(_SyncMemoryProvider)
    provider._backend = backend
    provider._runner = runner or Runner()
    provider._user_id = "u"
    provider._agent_id = "agent"
    provider._max_prefetch_chars = 2000
    provider._max_prefetch_timeout_ms = 3000
    provider._max_search_calls_per_turn = 3
    provider._search_calls_this_turn = 0
    provider._stable_profile = ""
    provider.name = "dual-mem"
    return provider


def test_system_prompt_block_stable_across_two_queries():
    class Backend:
        async def search(self, **kwargs):
            q = kwargs["query"]
            return SearchResult(
                success=True,
                request_id="req",
                memories=SearchMemories(
                    normal=[
                        MemoryItem(
                            memory_id=q,
                            content=f"动态命中:{q}",
                            category="fact",
                            score=0.8,
                        )
                    ],
                ),
                processing_time_ms=1.0,
            )

    provider = _provider_with_backend(Backend())
    provider._stable_profile = "<user-profile>\n- 姓名: 张三\n</user-profile>"
    first = provider.system_prompt_block()
    provider.prefetch("咖啡")
    mid = provider.system_prompt_block()
    provider.prefetch("工作")
    second = provider.system_prompt_block()
    assert first == mid == second
    assert "张三" in first
    assert "动态命中" not in first


def test_prefetch_formats_normal_only():
    searches: list[dict] = []

    class Backend:
        async def search(self, **kwargs):
            searches.append(kwargs)
            return SearchResult(
                success=True,
                request_id="req",
                memories=SearchMemories(
                    profile=[
                        MemoryItem(
                            memory_id="p1",
                            content="用户叫张三",
                            category="profile",
                            score=0.95,
                        )
                    ],
                    normal=[
                        MemoryItem(
                            memory_id="m1",
                            content="用户喜欢咖啡",
                            category="fact",
                            score=0.8,
                        )
                    ],
                ),
                processing_time_ms=1.0,
            )

    provider = _provider_with_backend(Backend())
    result = provider.prefetch("咖啡")
    assert "用户喜欢咖啡" in result
    assert "用户叫张三" not in result
    assert searches[0]["query"] == "咖啡"


def test_system_prompt_block_includes_cached_profile_and_tool_guide():
    provider = _provider_with_backend(object())
    provider._stable_profile = "<user-profile>\n用户叫张三\n</user-profile>"
    block = provider.system_prompt_block()
    assert "用户叫张三" in block
    assert MEMORY_TOOLS_GUIDE in block
    assert "memory_search" in block


def test_memory_search_returns_limit_reached_after_three_calls():
    class Backend:
        async def search(self, **kwargs):
            item = MemoryItem(
                memory_id="m1", content="hit", category="fact", score=0.9,
            )
            return SearchResult(
                success=True,
                request_id="req",
                memories=SearchMemories(normal=[item]),
                processing_time_ms=1.0,
            )

    provider = _provider_with_backend(Backend())
    provider.prefetch("第一轮")
    for _ in range(3):
        out = provider._dispatch_tool("memory_search", {"query": "x"})
        assert out["status"] == "success"
    limited = provider._dispatch_tool("memory_search", {"query": "x"})
    assert limited["status"] == "limit_reached"
    assert "上限" in limited["hint"]


def test_prefetch_timeout_returns_empty():
    class Backend:
        async def search(self, **kwargs):
            await asyncio.sleep(1)
            return SearchResult(
                success=True,
                request_id="req",
                memories=SearchMemories(),
                processing_time_ms=1.0,
            )

    provider = _provider_with_backend(Backend())
    provider._max_prefetch_timeout_ms = 10
    assert provider.prefetch("慢查询") == ""


def test_rendered_memory_context_total_chars():
    ctx = RenderedMemoryContext(stable_block="ab", dynamic_block="cde", total_chars=5)
    assert ctx.total_chars == 5


async def test_load_profile_block_includes_stable_topic_catalog(tmp_storage, fake_embed):
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(),
    )
    profile = MemoryNode(
        content="姓名: 张三",
        layer=Layer.L0_BASIC_INFO,
        app_id=client.settings.default_app_id,
        user_id="u",
        status=MemoryStatus.ACTIVE,
    )
    profile.embedding = fake_embed.embed_sync(profile.content)
    fact = MemoryNode(
        content="用户在 A 公司做支付",
        layer=Layer.L2_FACT,
        app_id=client.settings.default_app_id,
        user_id="u",
        status=MemoryStatus.ACTIVE,
        tags=["工作", "支付"],
    )
    fact.embedding = fake_embed.embed_sync(fact.content)
    client.factory.vector.upsert([profile, fact])
    backend = MemoryBackend(client=client)
    first = await backend.load_profile_block(user_id="u")
    second = await backend.load_profile_block(user_id="u")
    assert first == second
    assert "<user-profile>" in first
    assert "张三" in first
    assert "<topic-catalog>" in first
    assert "工作" in first
    assert "支付" in first
    await client.aclose()


def test_conversation_search_tool_is_registered_and_limited():
    class Backend:
        async def search(self, **kwargs):
            item = MemoryItem(
                memory_id="m1", content="hit", category="fact", score=0.9,
            )
            return SearchResult(
                success=True,
                request_id="req",
                memories=SearchMemories(normal=[item]),
                processing_time_ms=1.0,
            )

        async def search_conversation(self, **kwargs):
            item = MemoryItem(
                memory_id="l1", content="原话：我爱咖啡", category="raw", score=0.8,
            )
            return SearchResult(
                success=True,
                request_id="req",
                memories=SearchMemories(normal=[item]),
                processing_time_ms=1.0,
            )

    provider = _provider_with_backend(Backend())
    names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert "conversation_search" in names
    provider.prefetch("第一轮")
    out = provider._dispatch_tool("conversation_search", {"query": "咖啡"})
    assert out["status"] == "success"
    assert out["memories"][0]["content"] == "原话：我爱咖啡"
    for _ in range(2):
        assert provider._dispatch_tool("memory_search", {"query": "x"})["status"] == "success"
    limited = provider._dispatch_tool("conversation_search", {"query": "x"})
    assert limited["status"] == "limit_reached"


async def test_load_profile_block_is_query_independent(tmp_storage, fake_embed):
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(),
    )
    node = MemoryNode(
        content="姓名: 张三",
        layer=Layer.L0_BASIC_INFO,
        app_id=client.settings.default_app_id,
        user_id="u",
        status=MemoryStatus.ACTIVE,
    )
    node.embedding = fake_embed.embed_sync(node.content)
    client.factory.vector.upsert([node])
    backend = MemoryBackend(client=client)
    first = await backend.load_profile_block(user_id="u")
    second = await backend.load_profile_block(user_id="u")
    assert first == second
    assert "张三" in first
    assert first.startswith("<user-profile>")
    await client.aclose()
