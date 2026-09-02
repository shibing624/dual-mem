"""P1-4: Reflector UpdateType 分类 — OVERRIDE/MERGE/SUPPLEMENT/TEMPORAL/NEGATE/CONFLICT。

The reconciler now emits an `update_type` field per ADD op; the writer worker stores it
(plus optional temporal_scope and negation flag) into node.custom so the reader can render
a node's relationship to its predecessors without re-running the LLM.
"""
from dual_mem.agent.reconciler import (
    ReconcileOp,
    Reconciler,
    fold_absorb_deletes,
    resolve_skip_targets,
)
from dual_mem.storage.vector_store import ChromaVectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


def _parse(data) -> list[ReconcileOp]:
    return Reconciler._parse_ops(data)


def test_parse_override_supersedes():
    ops = _parse({
        "updates": [
            {"reason": "moved", "ops": [
                {"op": "ADD", "content": "用户搬到上海", "layer": "L4_IDENTITY",
                 "supersedes": ["m_old"], "tags": ["居住"], "update_type": "OVERRIDE"}
            ]}
        ]
    })
    assert len(ops) == 1
    op = ops[0]
    assert op.update_type == "OVERRIDE"
    assert op.supersedes == ["m_old"]
    assert op.negation is False
    assert op.temporal_scope is None


def test_parse_supplement_default_no_supersedes():
    ops = _parse([{"op": "ADD", "content": "用户也喜欢茶", "layer": "L4_IDENTITY",
                   "supersedes": [], "tags": ["饮品"], "update_type": "SUPPLEMENT"}])
    assert ops[0].update_type == "SUPPLEMENT"
    assert ops[0].supersedes == []


def test_parse_temporal_with_scope():
    ops = _parse([{"op": "ADD", "content": "用户今天想吃辣", "layer": "L2_FACT",
                   "supersedes": [], "tags": ["饮食"],
                   "update_type": "TEMPORAL", "temporal_scope": "今天"}])
    assert ops[0].update_type == "TEMPORAL"
    assert ops[0].temporal_scope == "今天"


def test_parse_negate_marks_negation_flag():
    ops = _parse([{"op": "ADD", "content": "用户不再喜欢苹果", "layer": "L4_IDENTITY",
                   "supersedes": ["m_apple"], "tags": ["饮食"], "update_type": "NEGATE"}])
    assert ops[0].update_type == "NEGATE"
    assert ops[0].negation is True
    assert ops[0].supersedes == ["m_apple"]


def test_parse_conflict_keeps_both():
    ops = _parse([{"op": "ADD", "content": "用户说自己今年30", "layer": "L4_IDENTITY",
                   "supersedes": [], "tags": ["age"], "update_type": "CONFLICT"}])
    assert ops[0].update_type == "CONFLICT"
    assert ops[0].supersedes == []
    assert ops[0].negation is False


def test_parse_unknown_update_type_falls_back_to_empty():
    """LLM 输出畸形 update_type 时，降级为空串而不是丢 op。"""
    ops = _parse([{"op": "ADD", "content": "x", "layer": "L2_FACT",
                   "supersedes": [], "tags": [], "update_type": "weird"}])
    assert len(ops) == 1
    assert ops[0].update_type == ""
    assert ops[0].negation is False


def test_parse_missing_update_type_is_optional():
    """旧格式不含 update_type 时仍然能解析（向前兼容当前 prompt 演进期）。"""
    ops = _parse([{"op": "ADD", "content": "x", "layer": "L2_FACT",
                   "supersedes": [], "tags": []}])
    assert len(ops) == 1
    assert ops[0].update_type == ""


def test_parse_merge_keeps_supersedes():
    ops = _parse([{
        "op": "ADD",
        "content": "用户在 A 公司负责支付与风控",
        "layer": "L2_FACT",
        "supersedes": ["m_pay"],
        "tags": ["工作"],
        "update_type": "MERGE",
    }])
    assert ops[0].update_type == "MERGE"
    assert ops[0].supersedes == ["m_pay"]
    assert ops[0].negation is False


def test_non_destructive_strips_merge_supersedes():
    reconciler = Reconciler.__new__(Reconciler)
    ops = reconciler._apply_non_destructive([
        ReconcileOp(
            op="ADD",
            content="用户在 A 公司负责支付与风控",
            supersedes=["old_pay"],
            update_type="MERGE",
        )
    ])
    assert ops[0].supersedes == []
    assert ops[0].update_type == "SUPPLEMENT"


def test_fold_objective2_add_delete_becomes_merge_chain():
    """旧 Objective 2（ADD + DELETE 吸收）必须在代码里收成 MERGE 链，不能靠 LLM 自觉。"""
    ops = fold_absorb_deletes([
        ReconcileOp(
            op="ADD",
            content="用户在 A 公司负责支付与风控",
            layer="L2_FACT",
            supersedes=[],
            update_type="SUPPLEMENT",
        ),
        ReconcileOp(op="DELETE", memory_id="old_pay"),
    ])
    assert len(ops) == 1
    assert ops[0].op == "ADD"
    assert ops[0].update_type == "MERGE"
    assert ops[0].supersedes == ["old_pay"]


def test_fold_drops_redundant_delete_on_override():
    ops = fold_absorb_deletes([
        ReconcileOp(
            op="ADD",
            content="用户搬到上海",
            supersedes=["old_bj"],
            update_type="OVERRIDE",
        ),
        ReconcileOp(op="DELETE", memory_id="old_bj"),
    ])
    assert len(ops) == 1
    assert ops[0].update_type == "OVERRIDE"
    assert ops[0].supersedes == ["old_bj"]


def test_fold_keeps_orphan_delete_when_no_add():
    ops = fold_absorb_deletes([ReconcileOp(op="DELETE", memory_id="dup")])
    assert len(ops) == 1
    assert ops[0].op == "DELETE"
    assert ops[0].memory_id == "dup"


async def test_reconcile_empty_updates_becomes_skip_all(tmp_storage, fake_embed):
    store = ChromaVectorStore(tmp_storage)
    seed = MemoryNode(
        content="用户喜欢喝咖啡",
        layer=Layer.L4_IDENTITY,
        app_id="app",
        user_id="u",
        agent_id="ag",
        status=MemoryStatus.ACTIVE,
        is_latest=True,
    )
    seed.embedding = fake_embed.embed_sync(seed.content)
    store.upsert([seed])
    llm = FakeLLMClient(responses={"search_query": [], "reconcile": {"updates": []}})
    rec = Reconciler(
        llm=llm, embed=fake_embed, vector=store, weak_candidate_score=0.0,
    )
    ops = await rec.reconcile(
        new_memories=["用户喜欢喝咖啡"],
        new_memories_meta=[{
            "content": "用户喜欢喝咖啡",
            "layer": "L4_IDENTITY",
            "tags": ["drink"],
            "node_id": "fw_new",
        }],
        app_id="app",
        user_id="u",
        agent_id="ag",
        current_time="",
        exclude_ids=["fw_new"],
    )
    assert [op.op for op in ops] == ["SKIP"]


def test_parse_skip_by_memory_id_and_index():
    ops = _parse([
        {"op": "SKIP", "memory_id": "fw1", "reason": "already stored"},
        {"op": "SKIP", "new_index": 2, "content": "用户喜欢咖啡"},
    ])
    assert [op.op for op in ops] == ["SKIP", "SKIP"]
    assert ops[0].memory_id == "fw1"
    assert ops[1].new_index == 2
    assert ops[1].content == "用户喜欢咖啡"


def test_resolve_skip_all_when_no_target():
    ids = resolve_skip_targets(
        [ReconcileOp(op="SKIP")],
        new_node_ids=["fw1", "fw2"],
        new_contents=["a", "b"],
    )
    assert sorted(ids) == ["fw1", "fw2"]


def test_resolve_skip_maps_index_and_content():
    ids = resolve_skip_targets(
        [
            ReconcileOp(op="SKIP", new_index=1),
            ReconcileOp(op="SKIP", content="用户喜欢茶"),
        ],
        new_node_ids=["fw1", "fw2"],
        new_contents=["用户喜欢咖啡", "用户喜欢茶"],
    )
    assert sorted(ids) == ["fw1", "fw2"]


def test_parse_merge_without_supersedes_downgrades_to_supplement():
    ops = _parse([{
        "op": "ADD",
        "content": "用户负责风控",
        "layer": "L2_FACT",
        "supersedes": [],
        "tags": ["工作"],
        "update_type": "MERGE",
    }])
    assert ops[0].update_type == "SUPPLEMENT"
    assert ops[0].supersedes == []
