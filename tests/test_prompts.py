import dual_mem.agent.prompts as prompts


def test_extract_prompts_format():
    for tmpl in (prompts.EXTRACT_ZH, prompts.EXTRACT_EN, prompts.SUMMARY_ZH, prompts.SUMMARY_EN):
        out = tmpl.format(content="对话", current_time="2026-06-18T10:00:00", history_context="")
        assert "对话" in out


def test_search_query_prompts_format():
    for tmpl in (prompts.SEARCH_QUERY_ZH, prompts.SEARCH_QUERY_EN):
        out = tmpl.format(new_memories="1. 用户喜欢咖啡")
        assert "用户喜欢咖啡" in out


def test_reconcile_prompts_format():
    for tmpl in (prompts.RECONCILE_ZH, prompts.RECONCILE_EN):
        out = tmpl.format(
            current_time="2026-06-18",
            existing_memories="[]",
            new_memories="1. 用户搬到上海",
            existing_tags="work, food",
        )
        assert "用户搬到上海" in out


def test_pick_switches_by_language():
    assert prompts.pick("ZH", "EN", "用户喜欢喝咖啡和茶") == "ZH"
    assert prompts.pick("ZH", "EN", "the user likes coffee") == "EN"
