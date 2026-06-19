"""P1-1 子PR2: FusionScorer W(d) — 各维度组合评分 + RRF 多路融合 + access freq。"""
import time
from datetime import timedelta

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.anchor_search import (
    PATH_SCHEMA,
    PATH_SEMANTIC,
    AnchorNode,
)
from dual_mem.retrieval.fusion_scorer import FusionConfig, FusionScorer
from dual_mem.types import Layer, MemoryNode, MemoryStatus


def _node(content: str, *, gmt: int | None = None, custom: dict | None = None,
          node_id: str | None = None, layer: Layer = Layer.L2_FACT) -> MemoryNode:
    n = MemoryNode(
        content=content,
        layer=layer,
        app_id="app",
        user_id="u",
        status=MemoryStatus.ACTIVE,
        gmt_created=gmt or int(time.time()),
        custom=custom,
    )
    if node_id:
        n.node_id = node_id
    return n


def test_fusion_scorer_basic_ordering(tmp_storage, fake_embed):
    """多锚点 → 按 final_score 降序排列；同节点多路径会合并。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    n1 = _node("a", node_id="n1")
    n2 = _node("b", node_id="n2")

    anchors = [
        AnchorNode(node=n1, score=0.9, source_path=PATH_SEMANTIC),
        AnchorNode(node=n2, score=0.5, source_path=PATH_SEMANTIC),
    ]
    scorer = FusionScorer(cache=factory.cache)
    ranked = scorer.score_and_rank(anchors=anchors)
    assert ranked[0].node_id == "n1"
    assert ranked[1].node_id == "n2"
    assert ranked[0].final_score > ranked[1].final_score


def test_fusion_access_count_lifts_rank(tmp_storage, fake_embed):
    """同分两节点：access_count 高的应排前面。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    n1 = _node("a", node_id="n1")
    n2 = _node("b", node_id="n2")
    # n2 被访问 50 次
    factory.cache.bump_access(["n2"] * 50)

    anchors = [
        AnchorNode(node=n1, score=0.7, source_path=PATH_SEMANTIC),
        AnchorNode(node=n2, score=0.7, source_path=PATH_SEMANTIC),
    ]
    scorer = FusionScorer(cache=factory.cache)
    ranked = scorer.score_and_rank(anchors=anchors)
    # n2 frequency 显著高，应排第一
    assert ranked[0].node_id == "n2"
    assert ranked[0].frequency > ranked[1].frequency


def test_fusion_time_decay_old_loses(tmp_storage, fake_embed):
    """同分两节点：旧节点（时间衰减 e^-λΔdays）排后。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    now = int(time.time())
    fresh = _node("fresh", node_id="fresh", gmt=now)
    old = _node("old", node_id="old", gmt=now - int(timedelta(days=365).total_seconds()))

    anchors = [
        AnchorNode(node=fresh, score=0.7, source_path=PATH_SEMANTIC),
        AnchorNode(node=old, score=0.7, source_path=PATH_SEMANTIC),
    ]
    scorer = FusionScorer(cache=factory.cache)
    ranked = scorer.score_and_rank(anchors=anchors)
    assert ranked[0].node_id == "fresh"
    assert ranked[0].time_decay > ranked[1].time_decay


def test_fusion_arousal_lifts_emotional_content(tmp_storage, fake_embed):
    """custom.emotional_arousal 高的节点 final_score 提升。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    flat = _node("flat", node_id="flat", custom={})
    excited = _node("excited", node_id="excited", custom={"emotional_arousal": 0.9})

    anchors = [
        AnchorNode(node=flat, score=0.7, source_path=PATH_SEMANTIC),
        AnchorNode(node=excited, score=0.7, source_path=PATH_SEMANTIC),
    ]
    scorer = FusionScorer(cache=factory.cache)
    ranked = scorer.score_and_rank(anchors=anchors)
    assert ranked[0].node_id == "excited"


def test_fusion_rrf_multi_path_boost(tmp_storage, fake_embed):
    """同节点出现在多条路径 → RRF 项加分。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    multi = _node("multi", node_id="multi", layer=Layer.L6_SCHEMA)
    only_one = _node("only", node_id="only")

    anchors = [
        AnchorNode(node=multi, score=0.6, source_path=PATH_SEMANTIC),
        AnchorNode(node=multi, score=0.6, source_path=PATH_SCHEMA),  # 同节点二次出现
        AnchorNode(node=only_one, score=0.7, source_path=PATH_SEMANTIC),
    ]
    scorer = FusionScorer(cache=factory.cache, config=FusionConfig(rrf_weight=1.0))
    ranked = scorer.score_and_rank(anchors=anchors, activated_schema_ids={"multi"})
    by_id = {s.node_id: s for s in ranked}
    # multi 节点 source_paths 应包含两条
    assert set(by_id["multi"].source_paths) == {PATH_SEMANTIC, PATH_SCHEMA}
    # multi 拿到 RRF 加成 + schema_boost
    assert by_id["multi"].rrf_score > by_id["only"].rrf_score
    assert by_id["multi"].schema_boost > 1.0


def test_fusion_empty_anchors():
    scorer = FusionScorer()
    assert scorer.score_and_rank(anchors=[]) == []
