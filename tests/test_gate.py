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


async def test_gate_reject_shortcircuit_skips_llm_on_near_duplicate():
    """Near-duplicate (very low vector novelty) + no biographical relevance → REJECT, no LLM."""
    llm = FakeLLMClient()
    gate = AttentionalGate(threshold=0.3, llm=llm)
    # max_sim 0.99 -> novelty ~0.04, below default reject threshold 0.12; content is low-value.
    sims = [{"node_id": "dup", "score": 0.99}]
    result = await gate.evaluate(content="嗯好的收到", existing_similarities=sims)
    assert result.passed is False
    assert result.scoring_method == "heuristic_shortcircuit_reject"
    assert llm.calls == []


async def test_gate_reject_shortcircuit_keeps_biographical_duplicate():
    """A near-duplicate that still looks biographical falls through to the LLM (not dropped)."""
    llm = FakeLLMClient(
        responses={
            "gate": {
                "novelty": 0.1,
                "biographical_relevance": 0.9,
                "emotional_arousal": 0.1,
                "reason": "profile",
            }
        }
    )
    gate = AttentionalGate(threshold=0.3, llm=llm)
    sims = [{"node_id": "dup", "score": 0.99}]
    result = await gate.evaluate(
        content="我叫张三，今年三十岁，在腾讯工作，住在深圳", existing_similarities=sims
    )
    assert result.scoring_method == "llm"
    assert len(llm.calls) == 1


async def test_gate_reject_shortcircuit_disabled_by_config():
    from dual_mem.agent.gate import GateConfig

    llm = FakeLLMClient(
        responses={"gate": {"novelty": 0.1, "biographical_relevance": 0.0, "emotional_arousal": 0.0}}
    )
    gate = AttentionalGate(
        llm=llm,
        config=GateConfig(threshold=0.3, shortcircuit_reject_novelty=0.0),
    )
    sims = [{"node_id": "dup", "score": 0.99}]
    await gate.evaluate(content="嗯好的收到", existing_similarities=sims)
    assert len(llm.calls) == 1  # reject short-circuit off → LLM still called
