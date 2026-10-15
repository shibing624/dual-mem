"""Public SDK coding flow preserves tool evidence and reports typed decisions."""

import pytest
from conftest import FakeJev, FakeLLMClient
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.coding.types import CodingDecisionError


def messages():
    return [
        {"role": "system", "content": "Host secret instructions"},
        {"role": "user", "content": "Fix the SDK transaction"},
        {
            "role": "tool",
            "content": "Verified transaction uses BEGIN IMMEDIATE",
            "tool_call_id": "tx",
        },
    ]


async def test_public_coding_returns_raw_and_decisions(tmp_storage, fake_embed):
    jev = FakeJev(
        {
            "scene": "CODING",
            "coverage": "VALID",
            "0:action": "ADD_SUPPLEMENT",
            "0:target": "NONE",
        }
    )
    llm = FakeLLMClient(
        {
            "json": [
                {
                    "task": "How to publish transactions",
                    "solution": "Use BEGIN IMMEDIATE to validate before commit",
                    "search_keys": [],
                }
            ]
        }
    )
    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage, coding_enabled=True),
        embed=fake_embed,
        llm=llm,
        jev=jev,
    )
    original = messages()
    r = await c.add(messages=original, user_id="u")
    assert r.success and r.commit_passed and r.extracted_count == 1
    assert r.memory_id and r.memory_id != r.extra_node_ids[0]
    assert r.decisions[0].action == "ADD" and r.decisions[0].update_type == "SUPPLEMENT"
    assert r.decisions[0].targets == [] and r.decisions[0].proposal_ids == ["p0"]
    assert r.decisions[0].source_ids == [r.memory_id]
    raw = await c.get(r.memory_id)
    assert "BEGIN IMMEDIATE" in raw.content and "Host secret" not in raw.content
    assert original[0]["content"] == "Host secret instructions"
    assert len(jev.calls) == 2 and len(llm.calls) == 1
    await c.aclose()


async def test_public_coding_generation_failure_preserves_tool_source(
    tmp_storage, fake_embed
):
    def failure(**kwargs):
        raise RuntimeError("generator unavailable")

    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage, coding_enabled=True),
        embed=fake_embed,
        llm=FakeLLMClient({"json": failure}),
        jev=FakeJev({"scene": "CODING"}),
    )
    with pytest.raises(RuntimeError, match="generator unavailable"):
        await c.add(messages=messages(), user_id="u")
    raws = c.factory.vector.get_many({"layer": "L0_RAW"})
    assert len(raws) == 1 and "BEGIN IMMEDIATE" in raws[0].content
    await c.aclose()
