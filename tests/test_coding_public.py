"""Public SDK coding flow preserves tool evidence and reports typed decisions."""

import asyncio
import json

import httpx
import respx
import pytest
from conftest import FakeJev, FakeLLMClient
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.coding.extractor import CodingMemoryExtractor
from dual_mem.providers.llm import LLMClient

CANDIDATE = {
    "task": "How to publish transactions",
    "solution": "Use BEGIN IMMEDIATE to validate before commit",
    "search_keys": [],
}


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


def coding_client(tmp_storage, fake_embed, jev, llm=None):
    return MemoryClient(
        settings=Settings(storage_dir=tmp_storage, coding_enabled=True),
        embed=fake_embed,
        llm=llm or FakeLLMClient({"json": {"candidates": [CANDIDATE]}}),
        jev=jev,
    )


async def test_public_coding_returns_raw_and_decisions(tmp_storage, fake_embed):
    jev = FakeJev(
        {
            "scene": "CODING",
            "0:action": "ADD_SUPPLEMENT",
            "0:target": "NONE",
        }
    )
    llm = FakeLLMClient({"json": {"candidates": [CANDIDATE]}})
    c = coding_client(tmp_storage, fake_embed, jev, llm)
    original = messages()
    r = await c.add(messages=original, user_id="u")
    assert r.success and r.commit_passed and r.extracted_count == 1
    assert r.memory_id and r.memory_id != r.extra_node_ids[0]
    assert r.decisions[0].action == "ADD" and r.decisions[0].update_type == "SUPPLEMENT"
    assert r.decisions[0].targets == [] and r.decisions[0].proposal_ids == ["p0"]
    assert r.decisions[0].source_ids == [r.memory_id]
    raw = await c.get(r.memory_id)
    assert "BEGIN IMMEDIATE" in raw.content and "Host secret" not in raw.content
    hits = await c.search_coding(query="publish transactions", user_id="u")
    assert len(hits) == 1 and hits[0]["memory_id"] == r.extra_node_ids[0]
    assert hits[0]["solution"] == "Use BEGIN IMMEDIATE to validate before commit"
    assert original[0]["content"] == "Host secret instructions"
    assert len(jev.calls) == 2 and len(llm.calls) == 1
    await c.aclose()


async def test_public_coding_generation_failure_preserves_tool_source(
    tmp_storage, fake_embed
):
    def failure(**kwargs):
        raise RuntimeError("generator unavailable")

    c = coding_client(
        tmp_storage,
        fake_embed,
        FakeJev({"scene": "CODING"}),
        FakeLLMClient({"json": failure}),
    )
    with pytest.raises(RuntimeError, match="generator unavailable"):
        await c.add(messages=messages(), user_id="u")
    raws = c.factory.vector.get_many({"layer": "L0_RAW"})
    assert len(raws) == 1 and "BEGIN IMMEDIATE" in raws[0].content
    await c.aclose()


@respx.mock
async def test_extractor_works_with_real_client_in_json_object_mode():
    """JSON-object mode (the default) can only return an object, never a bare array."""
    route = respx.post("https://api.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"candidates": [CANDIDATE]}),
                        },
                    }
                ],
            },
        )
    )
    llm = LLMClient(base_url="https://api.test/v1", api_key="k", model="m")
    drafts = await CodingMemoryExtractor(llm=llm).extract(messages(), user_id="u")
    body = json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert [(d.task, d.solution) for d in drafts] == [
        (CANDIDATE["task"], CANDIDATE["solution"])
    ]
    await llm.aclose()


async def test_concurrent_same_user_coding_writes_are_serialized(
    tmp_storage, fake_embed
):
    """Both writes publish; the second sees the first as a current head."""
    seen_heads: list[list[str]] = []

    def decide(*, state, questions):
        if "scene" not in questions:
            seen_heads.append([m["memory_id"] for m in state["current"]])
        return {"scene": "CODING", "0:target": "NONE"}

    c = coding_client(tmp_storage, fake_embed, FakeJev(decide))
    first, second = await asyncio.gather(
        c.add(messages=messages(), user_id="u"),
        c.add(messages=messages(), user_id="u"),
    )
    assert first.commit_passed and second.commit_passed
    assert seen_heads[0] == [] and seen_heads[1] == first.extra_node_ids
    await c.aclose()


async def test_structured_content_is_kept_as_json_not_python_repr(
    tmp_storage, fake_embed
):
    parts = [{"type": "text", "text": "Fix the SDK transaction"}]
    c = coding_client(
        tmp_storage,
        fake_embed,
        FakeJev({"scene": "CODING", "0:action": "ADD_SUPPLEMENT"}),
    )
    r = await c.add(
        messages=[{"role": "user", "content": parts}, *messages()[2:]], user_id="u"
    )
    source = c._coding_writer.store.get_source(r.memory_id, user_id="u")
    assert source["messages"][0]["content"] == parts
    await c.aclose()


@pytest.mark.parametrize(
    "action,target,passed,status",
    [
        ("DELETE", "head", True, "committed"),
        ("SKIP", "head", False, "known"),
    ],
)
async def test_commit_passed_matches_chat_contract(
    tmp_storage, fake_embed, action, target, passed, status
):
    """commit_passed = a structured version was appended OR a head was archived."""
    c = coding_client(
        tmp_storage,
        fake_embed,
        FakeJev({"scene": "CODING", "0:action": "ADD_SUPPLEMENT"}),
    )
    head = (await c.add(messages=messages(), user_id="u")).extra_node_ids[0]
    c._coding_writer.jev.responses = c._coding_writer.reconciler.jev.responses = {
        "scene": "CODING",
        "0:target": head if target == "head" else target,
        "0:action": action,
    }
    r = await c.add(messages=messages(), user_id="u")
    assert (r.commit_passed, r.status) == (passed, status)
    assert r.extracted_count == 0
    await c.aclose()
