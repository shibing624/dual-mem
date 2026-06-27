"""P1-4 完整闭环：ReconcilerWorker 应用 5 类 UpdateType 的 apply 行为单测。

Parsing 层已在 test_reflector 覆盖；这里测的是 worker 把 ReconcileOp 写到 vector store 后，
node.custom 是否带上 update_type / temporal_scope / negation 等元数据。下游读侧（Fusion /
ContextAssembly / 上层 prompt）依赖这些字段做消歧展示。
"""
import pytest

from dual_mem.agent.reconciler import ReconcileOp
from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.system2.reconciler_worker import ReconcilerWorker
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from tests.conftest import FakeLLMClient


@pytest.fixture
def worker_factory(tmp_storage, fake_embed):
    settings = Settings(mode="system1", storage_dir=tmp_storage, gate_enabled=False)
    return ComponentFactory(settings=settings, embed=fake_embed, llm=FakeLLMClient())


def _seed_existing(factory, node_id: str, content: str, layer: Layer = Layer.L4_IDENTITY) -> None:
    n = MemoryNode(
        content=content, layer=layer, app_id="app", user_id="u",
        status=MemoryStatus.ACTIVE, is_latest=True, node_id=node_id,
    )
    n.embedding = factory.embed.embed_sync(content)
    factory.vector.upsert([n])


async def _apply_one(factory, op: ReconcileOp) -> str | None:
    """Run worker._apply_ops on a single op and return the new node id (None for DELETE)."""
    worker = ReconcilerWorker(factory=factory)
    await worker._apply_ops(
        [op], app_id="app", user_id="u", agent_id="", session_id="s1",
    )
    # find the newest active node added (best-effort by content match).
    if op.op == "DELETE" or not op.content:
        return None
    nodes = factory.vector.get_many({"$and": [
        {"app_id": "app"}, {"user_id": "u"}, {"status": "ACTIVE"}
    ]}, limit=20)
    for n in nodes:
        if n.content == op.content:
            return n.node_id
    return None


async def test_apply_override_writes_update_type(worker_factory):
    """OVERRIDE op → new node.custom.update_type='OVERRIDE'，old 节点变 SUPERSEDED。"""
    _seed_existing(worker_factory, "old1", "用户住在北京")
    op = ReconcileOp(
        op="ADD", content="用户搬到了上海", layer="L4_IDENTITY",
        supersedes=["old1"], tags=["居住"], update_type="OVERRIDE",
    )
    new_id = await _apply_one(worker_factory, op)
    assert new_id is not None

    new_node = worker_factory.vector.get(new_id)
    assert new_node.custom and new_node.custom.get("update_type") == "OVERRIDE"
    # negation 不该被写
    assert "negation" not in (new_node.custom or {})

    old = worker_factory.vector.get("old1")
    assert old.is_latest is False
    assert old.status is MemoryStatus.SUPERSEDED


async def test_reconcile_apply_preserves_memory_at_from_superseded(worker_factory):
    """Reconcile ADD keeps session memory_at (LME QA uses conversation dates)."""
    ts = 1_700_000_000
    _seed_existing(worker_factory, "fw1", "用户有1300个Instagram粉丝", Layer.L2_FACT)
    old = worker_factory.vector.get("fw1")
    old.memory_at = ts
    worker_factory.vector.upsert([old])

    op = ReconcileOp(
        op="ADD",
        content="用户有1300个Instagram粉丝",
        layer="L2_FACT",
        supersedes=["fw1"],
        tags=["social"],
        update_type="OVERRIDE",
    )
    worker = ReconcilerWorker(factory=worker_factory)
    await worker._apply_ops(
        [op],
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="s1",
        memory_at_by_content={"用户有1300个instagram粉丝": ts},
        fallback_memory_at=ts,
    )
    nodes = worker_factory.vector.get_many({"$and": [
        {"app_id": "app"}, {"user_id": "u"}, {"status": "ACTIVE"}
    ]}, limit=10)
    merged = [n for n in nodes if "1300" in n.content]
    assert merged
    assert merged[0].memory_at == ts


async def test_apply_negate_marks_negation_in_custom(worker_factory):
    """NEGATE op → new node.custom.negation=True + update_type='NEGATE'。"""
    _seed_existing(worker_factory, "apple_id", "用户喜欢苹果")
    op = ReconcileOp(
        op="ADD", content="用户不再喜欢苹果", layer="L4_IDENTITY",
        supersedes=["apple_id"], tags=["饮食"], update_type="NEGATE", negation=True,
    )
    new_id = await _apply_one(worker_factory, op)
    new_node = worker_factory.vector.get(new_id)
    assert new_node.custom.get("negation") is True
    assert new_node.custom.get("update_type") == "NEGATE"


async def test_apply_temporal_carries_scope(worker_factory):
    """TEMPORAL op → new node.custom.temporal_scope 写入。"""
    op = ReconcileOp(
        op="ADD", content="用户今天想吃辣", layer="L2_FACT",
        supersedes=[], tags=["饮食"],
        update_type="TEMPORAL", temporal_scope="今天", negation=False,
    )
    new_id = await _apply_one(worker_factory, op)
    new_node = worker_factory.vector.get(new_id)
    assert new_node.custom.get("temporal_scope") == "今天"
    assert new_node.custom.get("update_type") == "TEMPORAL"


async def test_apply_conflict_keeps_both_no_supersede(worker_factory):
    """CONFLICT op → 不 supersede 任何旧节点，update_type='CONFLICT' 写入新节点。"""
    _seed_existing(worker_factory, "claim1", "用户说自己今年30")
    op = ReconcileOp(
        op="ADD", content="用户说自己今年35", layer="L4_IDENTITY",
        supersedes=[], tags=["age"], update_type="CONFLICT",
    )
    new_id = await _apply_one(worker_factory, op)
    new_node = worker_factory.vector.get(new_id)
    assert new_node.custom.get("update_type") == "CONFLICT"
    # 旧节点未被取代
    old = worker_factory.vector.get("claim1")
    assert old.is_latest is True
    assert old.status is MemoryStatus.ACTIVE


async def test_apply_supplement_no_metadata_pollution(worker_factory):
    """SUPPLEMENT op → custom 不会写入 negation/temporal_scope（避免读侧误判）。"""
    op = ReconcileOp(
        op="ADD", content="用户也喜欢茶", layer="L4_IDENTITY",
        supersedes=[], tags=["饮品"], update_type="SUPPLEMENT",
    )
    new_id = await _apply_one(worker_factory, op)
    new_node = worker_factory.vector.get(new_id)
    custom = new_node.custom or {}
    assert custom.get("update_type") == "SUPPLEMENT"
    assert "negation" not in custom
    assert "temporal_scope" not in custom
