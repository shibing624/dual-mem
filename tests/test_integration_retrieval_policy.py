"""Integration recall policy: every non-empty query reaches retrieval."""
import asyncio
from types import SimpleNamespace

from dual_mem.integrations._base import _SyncMemoryProvider
from dual_mem.integrations.agentica import DualMemWorkspace
from dual_mem.sdk_models import MemoryItem, SearchMemories, SearchResult


async def test_agentica_short_nonempty_query_is_retrieved() -> None:
    workspace = object.__new__(DualMemWorkspace)
    calls: list[tuple[str, int]] = []

    async def search(query: str, limit: int) -> list:
        calls.append((query, limit))
        return [SimpleNamespace(content="用户喜欢咖啡")]

    workspace._search = search
    workspace._last_recall = []
    workspace._build_entries = lambda items: []
    workspace._format_recall = lambda items, query, header: "recalled"

    result = await workspace.get_relevant_memories("好", limit=3)

    assert result == "recalled"
    assert calls == [("好", 3)]

    blank_result = await workspace.get_relevant_memories(" \t ", limit=3)
    assert blank_result == ""
    assert calls == [("好", 3)]


def test_sync_provider_short_nonempty_query_is_retrieved() -> None:
    searches: list[dict] = []

    class Backend:
        async def search(self, **kwargs):
            searches.append(kwargs)
            item = MemoryItem(
                memory_id="m1",
                content="用户喜欢咖啡",
                category="L2_FACT",
                score=0.9,
            )
            return SearchResult(
                success=True,
                request_id="req",
                memories=SearchMemories(normal=[item]),
                processing_time_ms=1.0,
            )

    class Runner:
        def run(self, coro):
            return asyncio.run(coro)

    provider = object.__new__(_SyncMemoryProvider)
    provider._backend = Backend()
    provider._runner = Runner()
    provider._user_id = "u"
    provider._agent_id = "agent"
    provider._max_prefetch_chars = 2000

    result = provider.prefetch("好")

    assert "用户喜欢咖啡" in result
    assert searches[0]["query"] == "好"

    blank_result = provider.prefetch(" \t ")
    assert blank_result == ""
    assert len(searches) == 1
