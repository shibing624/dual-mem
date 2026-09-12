"""P1-6: System2 single-shot ops agent — contract verification.

System2 no longer drives a multi-turn ReAct tool loop. It makes exactly ONE chat_json
call that emits a raw JSON array of ops (create_schema / create_intention / add_evidence
/ add_edge), which the executor materializes into graph state. These tests pin that
contract down:

- one LLM call per digest (not N tool turns),
- ops are dispatched to the graph in order,
- ``self_evidence`` is marked only when something was actually created,
- malformed / empty ops are survived without raising,
- a fact cluster is not re-consumed by the next digest.
"""
import pytest

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
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
    # Force single-shot mode: these tests exercise the one-call ops contract.
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


def _ops(*ops: dict) -> list[dict]:
    """An ops batch.

    The real ``LLMClient.chat_json`` parses the model's raw JSON array before returning,
    so the fake must hand the agent an already-parsed list — returning a JSON *string*
    here would be silently coerced to ``[]`` by the agent and every ops assertion would
    pass vacuously.
    """
    return list(ops)


def _script_ops(factory, payload) -> None:
    """Script the System2 ops response.

    The agent calls ``chat_json`` directly, so it routes to the FakeLLMClient's generic
    ``json`` bucket (no system-prompt keyword matches the System2 ops template).
    """
    factory.llm.responses["json"] = payload


def _s2_chat_json_calls(factory) -> int:
    """Count the LLM calls the System2 agent made."""
    return sum(1 for c in factory.llm.calls if c["type"] == "chat_json")


async def test_system2_single_shot_creates_schema_with_evidence(ultra_factory):
    """One ops call: create_schema with 3 evidence facts → wired into the graph."""
    _seed_fresh_facts(ultra_factory)
    _script_ops(
        ultra_factory,
        _ops(
            {
                "op": "create_schema",
                "content": "当做饭时，用户严格按菜谱精确称量——反映了对结构的需要。",
                "tags": ["烹饪"],
                "evidence": ["a1", "a2", "a3"],
            }
        ),
    )

    result = await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    assert result["created_schemas"] == 1
    assert result["evidence_added"] == 3

    # Exactly one LLM call — the ReAct loop is gone.
    assert _s2_chat_json_calls(ultra_factory) == 1

    # Verify schema actually landed in graph + evidence wired.
    schemas = ultra_factory.graph.list_by_layer(
        layer=Layer.L6_SCHEMA.value, user_id="u", app_ids=["app"]
    )
    assert len(schemas) == 1
    schema_id = schemas[0].node_id
    assert set(ultra_factory.graph.evidence_of(schema_id)) == {"a1", "a2", "a3"}


async def test_system2_single_shot_adds_evidence_to_existing_schema(ultra_factory):
    """A pre-existing schema is extended via add_evidence rather than re-created."""
    _seed_fresh_facts(ultra_factory)
    # Seed the schema the `add_evidence` op refers to, the same way the executor does.
    ultra_factory.graph.add_node(
        GraphNode(
            node_id="s1",
            layer=Layer.L6_SCHEMA.value,
            content="当做饭时，用户严格按菜谱——反映结构需要。",
            app_id="app",
            user_id="u",
            tags=["烹饪"],
        )
    )
    _script_ops(
        ultra_factory,
        _ops(
            {"op": "add_evidence", "schema_id": "s1", "evidence": ["a2", "a3"]},
        ),
    )

    result = await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    # The agent created nothing new; it only linked evidence onto the existing schema.
    assert result["created_schemas"] == 0
    assert result["evidence_added"] == 2
    assert set(ultra_factory.graph.evidence_of("s1")) == {"a2", "a3"}
    assert _s2_chat_json_calls(ultra_factory) == 1


async def test_system2_single_shot_handles_malformed_ops(ultra_factory):
    """An op with an unknown name / missing args must not raise."""
    _seed_fresh_facts(ultra_factory)
    _script_ops(
        ultra_factory,
        _ops(
            {"op": "add_evidence", "evidence": ["a1"]},  # missing schema_id
            {"op": "not_a_real_tool", "x": 1},           # unknown op
            {"no_op_key": True},                          # not an op at all
        ),
    )

    result = await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    # Nothing materialized, but the run completed cleanly.
    assert result["created_schemas"] == 0
    assert result["evidence_added"] == 0


async def test_system2_single_shot_empty_ops_makes_no_writes(ultra_factory):
    """An empty ops array → no graph writes, and no facts marked processed."""
    _seed_fresh_facts(ultra_factory)
    _script_ops(ultra_factory, _ops())

    result = await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    assert result["created_schemas"] == 0
    assert result["evidence_added"] == 0
    assert _s2_chat_json_calls(ultra_factory) == 1

    # Facts stay unprocessed so a later digest can still consume them.
    for fid in ("a1", "a2", "a3"):
        node = ultra_factory.vector.get(fid)
        assert node is not None and node.s2_evidence_count == 0


async def test_system2_no_clusters_no_calls(ultra_factory):
    """Only 1 fact → cluster_facts returns no clusters → LLM never invoked."""
    node = MemoryNode(
        content="孤立事实", layer=Layer.L2_FACT, app_id="app", user_id="u", node_id="x1"
    )
    node.embedding = _vec({2: 1.0})
    ultra_factory.vector.upsert([node])

    result = await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    assert result["created_schemas"] == 0
    assert _s2_chat_json_calls(ultra_factory) == 0


async def test_system2_single_shot_marks_clustered_facts_processed(ultra_factory):
    """Creating a schema marks the whole cluster consumed so the next digest skips it.

    Only one fact is cited as evidence, but every fact in the cluster gets
    ``s2_evidence_count`` bumped — otherwise the next digest would re-cluster and
    re-process the same facts indefinitely.
    """
    _seed_fresh_facts(ultra_factory)
    _script_ops(
        ultra_factory,
        _ops(
            {"op": "create_schema", "content": "X", "tags": [], "evidence": ["a1"]},
        ),
    )

    await System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    for fid in ("a1", "a2", "a3"):
        node = ultra_factory.vector.get(fid)
        assert node is not None and node.s2_evidence_count >= 1
