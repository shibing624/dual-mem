"""AttentionalGate: LLM-primary scoring with heuristic fallback."""

from dual_mem.agent.gate import AttentionalGate

from conftest import FakeLLMClient


async def test_gate_llm_primary_passes():
    llm = FakeLLMClient(
        responses={
            "gate": {
                "novelty": 0.9,
                "biographical_relevance": 0.8,
                "emotional_arousal": 0.2,
                "reason": "含用户偏好",
            },
        },
    )
    gate = AttentionalGate(threshold=0.3, llm=llm)
    result = await gate.evaluate(content="我比较喜欢川菜，但是花生过敏")
    assert result.scoring_method == "llm"
    assert result.passed is True
    assert result.gate_score > 0.3
    assert llm.calls[0]["type"] == "chat_json"


async def test_gate_llm_failure_falls_back_to_heuristic():
    llm = FakeLLMClient(responses={"gate": "not-a-dict"})

    async def _boom(**kw):
        raise RuntimeError("llm down")

    llm.chat_json = _boom  # type: ignore[method-assign]
    gate = AttentionalGate(threshold=0.3, llm=llm)
    result = await gate.evaluate(content="我是程序员，在腾讯工作")
    assert result.scoring_method == "heuristic"
    assert result.passed is True


async def test_gate_no_llm_uses_heuristic():
    gate = AttentionalGate(threshold=0.3, llm=None)
    result = await gate.evaluate(content="嗯嗯")
    assert result.scoring_method == "heuristic"
    assert result.passed is False


async def test_gate_rejects_empty():
    gate = AttentionalGate(threshold=0.3, llm=None)
    result = await gate.evaluate(content="   ")
    assert result.passed is False
    assert result.scoring_method == "rule"


async def test_gate_agent_context_injected_into_prompt():
    llm = FakeLLMClient()
    gate = AttentionalGate(threshold=0.3, llm=llm)
    await gate.evaluate(content="川菜", agent_context="你最喜欢什么菜系？")
    assert "你最喜欢什么菜系" in llm.calls[0]["system"]
