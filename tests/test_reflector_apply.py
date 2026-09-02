"""P1-4 完整闭环：ReconcilerWorker 应用 5 类 UpdateType 的 apply 行为单测。

Parsing 层已在 test_reflector 覆盖；这里测的是 worker 把 ReconcileOp 写到 vector store 后，
node.custom 是否带上 update_type / temporal_scope / negation 等元数据。下游读侧（Fusion /
ContextAssembly / 上层 prompt）依赖这些字段做消歧展示。
"""
import pytest

from dual_mem.agent.reconciler import ReconcileOp, Reconciler, fold_absorb_deletes
from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.system2.reconciler_worker import ReconcilerWorker
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from tests.conftest import FakeLLMClient


@pytest.fixture
def worker_factory(tmp_storage, fake_embed):
    settings = Settings(mode="system1", storage_dir=tmp_storage)
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


async def test_apply_folded_objective2_supersedes_instead_of_shadow(worker_factory):
    """ADD+DELETE 吸收经 fold 后走 SUPERSEDED 链，旧节点不是 SHADOW。"""
    _seed_existing(worker_factory, "old_pay", "用户在 A 公司做支付", Layer.L2_FACT)
    old = worker_factory.vector.get("old_pay")
    old.memory_at = 1_600_000_000
    worker_factory.vector.upsert([old])

    ops = fold_absorb_deletes([
        ReconcileOp(
            op="ADD",
            content="用户在 A 公司负责支付与风控",
            layer="L2_FACT",
            supersedes=[],
            tags=["工作"],
            update_type="SUPPLEMENT",
        ),
        ReconcileOp(op="DELETE", memory_id="old_pay"),
    ])
    worker = ReconcilerWorker(factory=worker_factory)
    await worker._apply_ops(
        ops,
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="s1",
        fallback_memory_at=1_700_000_000,
    )
    old = worker_factory.vector.get("old_pay")
    assert old.status is MemoryStatus.SUPERSEDED
    assert old.status is not MemoryStatus.SHADOW
    assert old.is_latest is False
    heads = [
        n for n in worker_factory.vector.get_many(
            {"$and": [{"app_id": "app"}, {"user_id": "u"}, {"status": "ACTIVE"}]},
            limit=10,
        )
        if "风控" in n.content
    ]
    assert heads
    assert (heads[0].custom or {}).get("update_type") == "MERGE"
    assert old.node_id in heads[0].supersedes


async def test_apply_merge_writes_merged_timestamps(worker_factory):
    """MERGE 才累积 custom.merged_timestamps；旧节点 SUPERSEDED 可回溯。"""
    old_ts = 1_600_000_000
    _seed_existing(worker_factory, "old_pay", "用户在 A 公司做支付", Layer.L2_FACT)
    old = worker_factory.vector.get("old_pay")
    old.memory_at = old_ts
    worker_factory.vector.upsert([old])

    new_ts = 1_700_000_000
    op = ReconcileOp(
        op="ADD",
        content="用户在 A 公司负责支付与风控",
        layer="L2_FACT",
        supersedes=["old_pay"],
        tags=["工作"],
        update_type="MERGE",
    )
    worker = ReconcilerWorker(factory=worker_factory)
    await worker._apply_ops(
        [op],
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="s1",
        fallback_memory_at=new_ts,
    )
    nodes = worker_factory.vector.get_many({"$and": [
        {"app_id": "app"}, {"user_id": "u"}, {"status": "ACTIVE"}
    ]}, limit=10)
    merged = [n for n in nodes if "风控" in n.content]
    assert merged
    stamps = (merged[0].custom or {}).get("merged_timestamps")
    assert stamps == [old_ts, new_ts]
    assert (merged[0].custom or {}).get("update_type") == "MERGE"
    old = worker_factory.vector.get("old_pay")
    assert old.status is MemoryStatus.SUPERSEDED
    assert old.is_latest is False


async def test_apply_merge_inherits_source_node_id(worker_factory):
    _seed_existing(worker_factory, "old_pay", "用户在 A 公司做支付", Layer.L2_FACT)
    op = ReconcileOp(
        op="ADD",
        content="用户在 A 公司负责支付与风控",
        layer="L2_FACT",
        supersedes=["old_pay"],
        tags=["工作"],
        update_type="MERGE",
    )
    worker = ReconcilerWorker(factory=worker_factory)
    await worker._apply_ops(
        [op],
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="s1",
        source_node_id="l1-raw",
    )
    nodes = worker_factory.vector.get_many({"$and": [
        {"app_id": "app"}, {"user_id": "u"}, {"status": "ACTIVE"}
    ]}, limit=10)
    merged = [n for n in nodes if "风控" in n.content]
    assert merged
    assert (merged[0].custom or {}).get("source_node_id") == "l1-raw"


async def test_apply_override_does_not_write_merged_timestamps(worker_factory):
    """OVERRIDE 不得写 merged_timestamps，避免替换链画出虚假『持续』区间。"""
    _seed_existing(worker_factory, "old_java", "用户主力语言是 Java", Layer.L4_IDENTITY)
    old = worker_factory.vector.get("old_java")
    old.memory_at = 1_600_000_000
    worker_factory.vector.upsert([old])

    op = ReconcileOp(
        op="ADD",
        content="用户主力语言是 Python",
        layer="L4_IDENTITY",
        supersedes=["old_java"],
        tags=["lang"],
        update_type="OVERRIDE",
    )
    worker = ReconcilerWorker(factory=worker_factory)
    await worker._apply_ops(
        [op],
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="s1",
        fallback_memory_at=1_700_000_000,
    )
    nodes = worker_factory.vector.get_many({"$and": [
        {"app_id": "app"}, {"user_id": "u"}, {"status": "ACTIVE"}
    ]}, limit=10)
    heads = [n for n in nodes if "Python" in n.content]
    assert heads
    assert "merged_timestamps" not in (heads[0].custom or {})


async def test_empty_updates_skip_all_retracts_fast_write(worker_factory, fake_embed):
    """LLM 明确空 updates = 无增量，必须收回 fast-write，不能让重复节点继续 ACTIVE。"""
    existing = MemoryNode(
        content="用户喜欢喝美式咖啡",
        layer=Layer.L2_FACT,
        app_id="app",
        user_id="u",
        status=MemoryStatus.ACTIVE,
        is_latest=True,
        node_id="existing",
    )
    existing.embedding = fake_embed.embed_sync(existing.content)
    worker_factory.vector.upsert([existing])
    _seed_existing(worker_factory, "fw_dup", "用户喜欢喝美式咖啡", Layer.L2_FACT)

    worker_factory.settings.reconcile_weak_candidate_score = 0.0
    llm = FakeLLMClient(responses={"reconcile": {"updates": []}})
    worker = ReconcilerWorker(factory=worker_factory)
    worker.reconciler = Reconciler(
        llm=llm,
        embed=fake_embed,
        vector=worker_factory.vector,
        weak_candidate_score=0.0,
    )
    worker_factory.cache.enqueue_reconcile_task(
        app_id="app", user_id="u", agent_id="", node_ids=["fw_dup"],
    )
    await worker.reconcile_pending(app_id="app", user_id="u", agent_id="")
    fw = worker_factory.vector.get("fw_dup")
    assert fw.status is MemoryStatus.SHADOW
    assert fw.is_latest is False
    kept = worker_factory.vector.get("existing")
    assert kept.status is MemoryStatus.ACTIVE


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
