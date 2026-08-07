from dual_mem.sdk_models import MemoryItem, SearchMemories


def test_search_memories_flatten_deduplicates_routes():
    shared = MemoryItem(
        memory_id="same",
        content="shared fact",
        category="fact",
        score=0.9,
    )
    memories = SearchMemories(profile=[shared], normal=[shared])

    assert memories.flatten() == [shared]