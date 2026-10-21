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


def test_flatten_schemas_never_take_evidence_slots():
    """Schemas are appended after the bounded evidence list, whatever their score."""
    facts = [
        MemoryItem(memory_id=f"f{i}", content="fact", category="fact", score=9 - i, layer="L1_FACT")
        for i in range(6)
    ]
    schemas = [
        MemoryItem(memory_id=f"s{i}", content="schema", category="schema", score=99 - i, layer="L3_DERIVED")
        for i in range(2)
    ]
    memories = SearchMemories(normal=facts, schema=schemas)
    result = [r.memory_id for r in memories.flatten(limit=4)]
    assert result == ["f0", "f1", "f2", "f3", "s0", "s1"]
    assert set(memories.to_dict()) == {"profile", "proactive", "normal", "raw", "schema", "catalog"}