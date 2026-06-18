from dual_mem.retrieval.formatter import format_memories


def test_format_groups_chain_and_raw_truncate():
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
                "evolution_chain": [
                    {"node_id": "a", "content": "最新偏好：喜欢咖啡"},
                    {"node_id": "b", "content": "旧版偏好：喜欢茶"},
                ],
            },
            {"memory_id": "r", "content": "原" * 1000, "category": "raw"},
        ],
    }

    text = format_memories(result, raw_truncate=800)

    # 分组标题（proactive 为空被跳过）
    assert "【画像 Profile】" in text
    assert "【常规记忆 Normal】" in text
    assert "【主动 Proactive】" not in text
    # 画像内容
    assert "用户叫张三" in text
    # 演化链多版本展开
    assert "演化历史" in text
    assert "最新偏好：喜欢咖啡" in text
    assert "旧版偏好：喜欢茶" in text
    # raw 截断到 800 字 + 省略号
    assert "原" * 800 + "…" in text
    assert "原" * 801 not in text


def test_format_empty_result_is_empty_string():
    assert format_memories({"profile": [], "proactive": [], "normal": []}) == ""
