"""P1-6: System2 ReAct agent — tool_calls 循环验证。

The new agent drives an OpenAI function-calling loop over the 8 tools. Tests scripts
sequences of FakeLLM tool_calls and assert the executor materializes them into graph
state correctly. Empty tool_calls terminates the loop.
"""
import json

import pytest

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.system2.system2_agent import System2Agent
from dual_mem.types import Layer, MemoryNode

from tests.conftest import FakeLLMClient


def _vec(comps: dict[int, float], dim: int = 16) -> list[float]:
    v = [0.0] * dim
    for i, val in comps.items():
        v[i] = val
    return v


@pytest.fixture
def ultra_factory(tmp_storage, fake_embed):
    # Disable single-shot so these tests exercise the multi-turn ReAct tool loop.
    settings = Settings(
        mode="dual", storage_dir=tmp_storage, system2_single_shot_max_clusters=0
    )
    return ComponentFactory(settings=settings, embed=fake_embed, llm=FakeLLMClient())


def _seed_fresh_facts(factory):
    nodes = []
    for nid, content, comp in [
        ("a1", "用户做饭严格按菜谱", {0: 1.0, 4: 0.5}),
        ("a2", "用户烘焙精确称量", {0: 1.0, 5: 0.5}),
        ("a3", "用户煮咖啡按固定比例", {0: 1.0, 6: 0.5}),
    ]:
        node = MemoryNode(
            content=content, layer=Layer.L2_FACT, app_id="app", user_id="u", node_id=nid
        )
        node.embedding = _vec(comp)
        nodes.append(node)
    factory.vector.upsert(nodes)


def _tool_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


async def test_system2_react_creates_schema_with_evidence(ultra_factory):
    """Scripted ReAct: create_schema → add_evidence (3 facts) → stop."""
    _seed_fresh_facts(ultra_factory)

    turns = [
        {
            "content": "I will create a schema for cooking precision.",
            "tool_calls": [
                _tool_call(
                    "tc1",
                    "create_schema",
                    {
                        "content": "当做饭时，用户严格按菜谱精确称量——反映了对结构的需要。",
                        "tags": ["烹饪"],
                        "evidence": ["a1", "a2", "a3"],
                    },
                )
            ],
        },
        {"content": "Done.", "tool_calls": []},
    ]
    ultra_factory.llm.responses["tools"] = turns

    result = await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    assert result["created_schemas"] == 1
    assert result["evidence_added"] == 3

    # Verify schema actually landed in graph + evidence wired.
    schemas = ultra_factory.graph.list_by_layer(
        layer=Layer.L6_SCHEMA.value, user_id="u", app_ids=["app"]
    )
    assert len(schemas) == 1
    schema_id = schemas[0].node_id
    assert set(ultra_factory.graph.evidence_of(schema_id)) == {"a1", "a2", "a3"}


async def test_system2_react_search_then_add_evidence(ultra_factory):
    """search_graph (empty) → create_schema → add_evidence (1 more fact, separate turn)."""
    _seed_fresh_facts(ultra_factory)

    turns = [
        {
            "content": "Searching first.",
            "tool_calls": [
                _tool_call("tc1", "search_graph",
                           {"query": "cooking precision", "layer": "L6_SCHEMA"})
            ],
        },
        {
            "content": "No existing schema; creating new.",
            "tool_calls": [
                _tool_call(
                    "tc2", "create_schema",
                    {"content": "当X时用户Y——反映Z。", "tags": [],
                     "evidence": ["a1", "a2"]}
                )
            ],
        },
        {"content": "Adding the third fact as evidence.", "tool_calls": []},
    ]
    ultra_factory.llm.responses["tools"] = turns

    result = await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    assert result["created_schemas"] == 1
    assert result["evidence_added"] == 2


async def test_system2_react_handles_malformed_args(ultra_factory):
    """LLM emits a malformed add_evidence (missing schema_id) → executor returns error,
    the loop survives and the next turn (empty) terminates cleanly."""
    _seed_fresh_facts(ultra_factory)

    turns = [
        {
            "content": "",
            "tool_calls": [
                _tool_call("tc1", "add_evidence", {"evidence": ["a1"]}),  # missing schema_id
            ],
        },
        {"content": "OK stopping.", "tool_calls": []},
    ]
    ultra_factory.llm.responses["tools"] = turns

    result = await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    # No schema created, no evidence linked, but no exception.
    assert result["created_schemas"] == 0
    assert result["evidence_added"] == 0


async def test_system2_react_stops_at_max_iters(ultra_factory):
    """If the LLM never stops, loop honors system2_max_iters."""
    _seed_fresh_facts(ultra_factory)
    ultra_factory.settings.system2_max_iters = 2

    def always_emit_a_call(*, messages, tools):
        return {
            "content": "",
            "tool_calls": [_tool_call("tc", "search_vdb", {"query": "x"})],
        }

    ultra_factory.llm.responses["tools"] = always_emit_a_call

    result = await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    # No write tools called → no schema/evidence
    assert result["created_schemas"] == 0
    # Verify we hit the cap (max_iters=2 means at most 2 chat_with_tools calls).
    tool_calls_count = sum(
        1 for c in ultra_factory.llm.calls if c["type"] == "chat_with_tools"
    )
    assert tool_calls_count == 2


async def test_system2_react_no_clusters_no_calls(ultra_factory):
    """Only 1 fact → cluster_facts returns no clusters → LLM never invoked."""
    node = MemoryNode(
        content="孤立事实", layer=Layer.L2_FACT, app_id="app", user_id="u", node_id="x1"
    )
    node.embedding = _vec({2: 1.0})
    ultra_factory.vector.upsert([node])

    result = await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    assert result["created_schemas"] == 0
    # LLM should not be called at all.
    assert all(c["type"] != "chat_with_tools" for c in ultra_factory.llm.calls)


async def test_system2_react_marks_clustered_facts_processed(ultra_factory):
    """Even if LLM creates schema with only 1 fact, the rest of the cluster gets
    s2_evidence_count bumped to 1 so a second digest does not re-process them."""
    _seed_fresh_facts(ultra_factory)

    turns = [
        {
            "content": "",
            "tool_calls": [
                _tool_call("tc1", "create_schema",
                           {"content": "X", "tags": [], "evidence": ["a1"]}),
            ],
        },
        {"content": "", "tool_calls": []},
    ]
    ultra_factory.llm.responses["tools"] = turns

    await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    for fid in ("a1", "a2", "a3"):
        node = ultra_factory.vector.get(fid)
        assert node is not None and node.s2_evidence_count >= 1
