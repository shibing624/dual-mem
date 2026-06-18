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
    settings = Settings(mode="ultra", storage_dir=tmp_storage)
    ops = [
        {
            "op": "create_schema",
            "content": "当做饭时，用户严格按菜谱精确称量——反映了用外部结构管理不确定性的需要。",
            "tags": ["烹饪"],
            "evidence": ["a1", "a2", "a3"],
        }
    ]
    llm = FakeLLMClient(responses={"json": ops})
    return ComponentFactory(settings=settings, embed=fake_embed, llm=llm)


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


def test_system2_agent_creates_schema_with_evidence(ultra_factory):
    _seed_fresh_facts(ultra_factory)

    result = System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    assert result["created_schemas"] == 1
    assert result["evidence_added"] == 3

    schemas = ultra_factory.graph.list_by_layer(
        layer=Layer.L6_SCHEMA.value, user_id="u", app_ids=["app"]
    )
    assert len(schemas) == 1
    schema_id = schemas[0].node_id

    assert set(ultra_factory.graph.evidence_of(schema_id)) == {"a1", "a2", "a3"}
    for fid in ("a1", "a2", "a3"):
        assert ultra_factory.vector.get(fid).s2_evidence_count == 1


def test_tool_executor_skips_malformed_ops(tmp_storage, fake_embed):
    """LLM 输出畸形 ops（缺字段/非法 rel/非 dict）不应让 digest 崩溃。"""
    settings = Settings(mode="ultra", storage_dir=tmp_storage)
    bad_ops = [
        {"op": "create_schema"},  # 缺 content
        {"op": "add_evidence", "evidence": ["a1"]},  # 缺 schema_id
        {"op": "add_edge", "from_id": "a1"},  # 缺 to_id
        "not-a-dict",  # 非 dict
        {"op": "create_schema", "content": "当X时用户Y——反映Z。", "tags": [], "evidence": ["a1", "a2", "a3"]},
    ]
    llm = FakeLLMClient(responses={"json": bad_ops})
    factory = ComponentFactory(settings=settings, embed=fake_embed, llm=llm)
    _seed_fresh_facts(factory)

    result = System2Agent(factory=factory).run(app_id="app", user_id="u")
    assert result["created_schemas"] == 1
    assert result["evidence_added"] == 3


def test_system2_marks_clustered_facts_processed(ultra_factory):
    """聚类中未被引用为 evidence 的 fact 也应标记已加工，避免重复消费。"""
    _seed_fresh_facts(ultra_factory)
    # ops 只引用 a1，a2/a3 未被引用但参与了聚类
    ultra_factory.llm.responses["json"] = [
        {"op": "create_schema", "content": "当做饭时用户精确称量——反映控制欲。", "tags": [], "evidence": ["a1"]}
    ]

    System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    for fid in ("a1", "a2", "a3"):
        assert ultra_factory.vector.get(fid).s2_evidence_count >= 1

    # 第二次 digest 不应再有 fresh fact → 不再造 schema
    result2 = System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    assert result2["created_schemas"] == 0


def test_system2_agent_no_clusters_no_ops(ultra_factory):
    # 只有 2 条 fact，不足以成簇 → 空 ops
    node = MemoryNode(
        content="孤立事实", layer=Layer.L2_FACT, app_id="app", user_id="u", node_id="x1"
    )
    node.embedding = _vec({2: 1.0})
    ultra_factory.vector.upsert([node])

    result = System2Agent(factory=ultra_factory).run(app_id="app", user_id="u")
    assert result["created_schemas"] == 0
