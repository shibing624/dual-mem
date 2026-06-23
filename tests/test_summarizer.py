from dual_mem.agent.summarizer import Summarizer

from conftest import FakeLLMClient


async def test_short_content_returns_none():
    llm = FakeLLMClient(responses={"text": "不应被调用"})
    summarizer = Summarizer(llm=llm)
    assert await summarizer.summarize(content="短文本", current_time="") is None
    assert llm.calls == []


async def test_long_content_calls_llm():
    long_text = "用户" + "聊了很多关于旅行和美食的事情。" * 110
    assert len(long_text) >= 1500
    llm = FakeLLMClient(responses={"text": "用户喜欢旅行和美食。"})
    summarizer = Summarizer(llm=llm)
    out = await summarizer.summarize(content=long_text, current_time="2026-06-18")
    assert out == "用户喜欢旅行和美食。"
    assert llm.calls[0]["type"] == "chat_text"


async def test_empty_summary_returns_none():
    long_text = "x" * 1500
    llm = FakeLLMClient(responses={"text": "   "})
    summarizer = Summarizer(llm=llm)
    assert await summarizer.summarize(content=long_text, current_time="") is None
