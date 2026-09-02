from dual_mem.integrations._base import format_memories_for_prompt
from dual_mem.retrieval.formatter import (
    format_evolution_timeline,
    format_memories,
    format_memory_timestamp,
    format_search_hit_line,
)
from dual_mem.sdk_models import EvolutionItem, MemoryItem, SearchMemories


def test_format_groups_chain_head_only_and_raw_truncate():
    result = {
        "profile": [
            {"memory_id": "p0", "content": "用户叫张三", "category": "profile"},
        ],
        "proactive": [],
        "normal": [
            {
                "memory_id": "a",
                "content": "最新偏好：喜欢咖啡",
                "category": "fact",
                "memory_at": 1_700_000_000,
                "evolution_chain": [
                    {"node_id": "a", "content": "最新偏好：喜欢咖啡"},
                    {"node_id": "b", "content": "旧版偏好：喜欢茶"},
                ],
            },
            {"memory_id": "r", "content": "原" * 1000, "category": "raw"},
        ],
    }

    text = format_memories(result, raw_truncate=800)

    assert "【画像 Profile】" in text
    assert "【常规记忆 Normal】" in text
    assert "【主动 Proactive】" not in text
    assert "用户叫张三" in text
    assert "最新偏好：喜欢咖啡" in text
    assert "conversation date:" in text
    assert "not today's date" in text
    # superseded hidden by default (knowledge-update fix)
    assert "旧版偏好：喜欢茶" not in text
    assert "演化历史" not in text
    assert "原" * 800 + "…" in text
    assert "原" * 801 not in text


def test_format_include_superseded():
    result = {
        "profile": [],
        "proactive": [],
        "normal": [
            {
                "memory_id": "a",
                "content": "Currently obsessed with Sweet Baby Ray's",
                "category": "fact",
                "evolution_chain": [
                    {"node_id": "a", "content": "Currently obsessed with Sweet Baby Ray's"},
                    {"node_id": "b", "content": "Likes Kansas City Masterpiece"},
                ],
            },
        ],
    }
    text = format_memories(result, include_superseded=True)
    assert "Sweet Baby Ray's" in text
    assert "Kansas City Masterpiece" in text
    assert "superseded" in text


def test_format_empty_result_is_empty_string():
    assert format_memories({"profile": [], "proactive": [], "normal": []}) == ""


def test_memory_item_to_search_result_omits_evolution():
    item = MemoryItem(
        memory_id="a",
        content="600 followers",
        category="fact",
        score=0.9,
        memory_at=1_700_000_000,
        evolution_chain=[
            EvolutionItem(node_id="a", content="600 followers", layer="L2_FACT"),
            EvolutionItem(node_id="b", content="500 followers", layer="L2_FACT"),
        ],
    )
    hit = item.to_search_result()
    assert hit["memory"] == "600 followers"
    assert "500" not in hit["memory"]
    assert hit["created_at"].startswith("2023-")


def test_search_memories_to_search_results():
    memories = SearchMemories(
        normal=[
            MemoryItem(memory_id="1", content="a", category="fact", score=0.5),
            MemoryItem(memory_id="2", content="b", category="fact", score=0.9),
        ],
    )
    hits = memories.to_search_results(limit=1)
    assert len(hits) == 1
    assert hits[0]["memory"] == "b"


def test_format_merge_shows_folded_facts_as_still_valid():
    result = {
        "profile": [],
        "proactive": [],
        "normal": [
            {
                "memory_id": "a",
                "content": "用户在 A 公司负责支付与风控",
                "category": "fact",
                "update_type": "MERGE",
                "merged_timestamps": [1_600_000_000, 1_700_000_000],
                "evolution_chain": [
                    {"node_id": "a", "content": "用户在 A 公司负责支付与风控"},
                    {"node_id": "b", "content": "用户在 A 公司做支付"},
                ],
            }
        ],
    }
    text = format_memories(result)
    assert "folded, still valid" in text
    assert "用户在 A 公司做支付" in text
    assert "not current" not in text


def test_format_merge_duration_when_timestamps_span():
    result = {
        "profile": [],
        "proactive": [],
        "normal": [
            {
                "memory_id": "a",
                "content": "用户在 A 公司负责支付与风控",
                "category": "fact",
                "memory_at": 1_700_000_000,
                "merged_timestamps": [1_600_000_000, 1_700_000_000],
            }
        ],
    }
    text = format_memories(result)
    assert "(持续: 2020-09-13 → 2023-11-14)" in text


def test_format_override_omits_duration_without_merged_timestamps():
    result = {
        "profile": [],
        "proactive": [],
        "normal": [
            {
                "memory_id": "a",
                "content": "用户主力语言是 Python",
                "category": "fact",
                "memory_at": 1_700_000_000,
                "evolution_chain": [
                    {"node_id": "a", "content": "用户主力语言是 Python", "memory_at": 1_700_000_000},
                    {"node_id": "b", "content": "用户主力语言是 Java", "memory_at": 1_600_000_000},
                ],
            }
        ],
    }
    text = format_memories(result)
    assert "持续" not in text


def test_format_memories_shows_source_node_id():
    result = {
        "profile": [],
        "proactive": [],
        "normal": [
            {
                "memory_id": "f1",
                "content": "用户喜欢咖啡",
                "category": "fact",
                "source_node_id": "l1-abcd1234eeee",
            }
        ],
    }
    text = format_memories(result, date_preamble=False)
    assert "来源: L1 l1-abcd1" in text


def test_format_memories_for_prompt_shows_source_node_id():
    item = MemoryItem(
        memory_id="f1",
        content="用户喜欢咖啡",
        category="fact",
        score=0.9,
        source_node_id="l1-abcd1234eeee",
    )
    text = format_memories_for_prompt([item])
    assert "来源: L1 l1-abcd1" in text


def test_format_memory_timestamp():
    assert format_memory_timestamp(None) == ""
    assert format_memory_timestamp(1_700_000_000) == "2023-11-14"


def test_format_search_hit_line_with_chain():
    hit = {
        "content": "Now likes tea",
        "category": "fact",
        "memory_at": 1_700_000_000,
        "evolution_chain": [
            {"node_id": "a", "content": "Now likes tea", "memory_at": 1_700_000_000},
            {"node_id": "b", "content": "Used to like coffee", "memory_at": 1_600_000_000},
        ],
    }
    text = format_search_hit_line(hit, include_evolution=True)
    assert "Now likes tea" in text
    assert "Used to like coffee" in text


def test_format_evolution_timeline():
    hits = [
        {
            "content": "Current",
            "score": 1.0,
            "evolution_chain": [
                {"node_id": "a", "content": "Current", "memory_at": 2},
                {"node_id": "b", "content": "Old", "memory_at": 1},
            ],
        }
    ]
    text = format_evolution_timeline(hits, header="Timeline:")
    assert "Timeline:" in text
    assert "Old" in text
    assert "CURRENT" in text
