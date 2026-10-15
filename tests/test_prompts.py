import dual_mem.agent.prompts as prompts


def test_extract_prompts_format():
    for tmpl in (
        prompts.EXTRACT_ZH,
        prompts.EXTRACT_EN,
    ):
        out = tmpl.format(content="对话", current_time="2026-06-18T10:00:00")
        assert "对话" in out


def test_pick_switches_by_language():
    assert prompts.pick("ZH", "EN", "用户喜欢喝咖啡和茶") == "ZH"
    assert prompts.pick("ZH", "EN", "the user likes coffee") == "EN"
