from dual_mem.agent.summarizer import Summarizer

from conftest import FakeLLMClient


def test_short_content_returns_none():
    llm = FakeLLMClient(responses={"text": "不应被调用"})
    summarizer = Summarizer(llm=llm)
    assert summarizer.summarize(content="短文本", current_time="") is None
    assert llm.calls == []


def test_long_content_calls_llm():
    long_text = "用户" + "聊了很多关于旅行和美食的事情。" * 60
    assert len(long_text) >= 500
    llm = FakeLLMClient(responses={"text": "用户喜欢旅行和美食。"})
    summarizer = Summarizer(llm=llm)
    out = summarizer.summarize(content=long_text, current_time="2026-06-18")
    assert out == "用户喜欢旅行和美食。"
    assert llm.calls[0]["type"] == "chat_text"


def test_empty_summary_returns_none():
    long_text = "x" * 600
    llm = FakeLLMClient(responses={"text": "   "})
    summarizer = Summarizer(llm=llm)
    assert summarizer.summarize(content=long_text, current_time="") is None
