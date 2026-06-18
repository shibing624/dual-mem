from datetime import datetime, timedelta

from dual_mem.retrieval.intent import (
    classify_intent,
    extract_keywords,
    is_conceptual,
    is_navigational,
    parse_time_range,
)


def test_navigational_signals():
    assert is_navigational("调用 `getCwd` 函数") is True
    assert is_navigational('看看 "exact phrase" 是啥') is True
    assert is_navigational("id 是 550e8400-e29b-41d4-a716-446655440000") is True
    assert classify_intent("`getCwd` 怎么用") == "NAVIGATIONAL"


def test_conceptual_intent():
    assert is_conceptual("为什么要这么设计") is True
    assert classify_intent("why do we do this") == "CONCEPTUAL"


def test_factual_default():
    assert classify_intent("用户喜欢喝咖啡") == "FACTUAL"


def test_extract_keywords_mixed():
    kws = extract_keywords("用户喜欢 Python 和 machine learning")
    assert "python" in kws
    assert "machine" in kws
    assert "用户喜欢" in kws
    assert all(len(k) >= 2 for k in kws)
    assert len(kws) == len(set(kws))


def test_extract_keywords_drops_stopwords_and_short():
    # 全部停用词或 <3 字符英文，应被过滤掉
    assert extract_keywords("the a is my go") == []


def test_parse_time_range_yesterday():
    now = datetime(2026, 6, 18, 15, 30, 0)
    ts = parse_time_range("昨天聊了啥", now=now)
    expected = datetime(2026, 6, 17, 0, 0, 0)
    assert ts == int(expected.timestamp())


def test_parse_time_range_recent_n_days():
    now = datetime(2026, 6, 18, 15, 30, 0)
    ts = parse_time_range("最近3天的对话", now=now)
    expected = datetime(2026, 6, 15, 0, 0, 0)
    assert ts == int(expected.timestamp())


def test_parse_time_range_none_when_no_time_word():
    assert parse_time_range("讲讲机器学习") is None
