"""P1-4: Reflector UpdateType 分类 — OVERRIDE/SUPPLEMENT/TEMPORAL/NEGATE/CONFLICT。

The reconciler now emits an `update_type` field per ADD op; the writer worker stores it
(plus optional temporal_scope and negation flag) into node.custom so the reader can render
a node's relationship to its predecessors without re-running the LLM.
"""
from dual_mem.agent.reconciler import ReconcileOp, Reconciler


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
