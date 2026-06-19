"""P1-1 子PR3: GraphExpander — 1-hop 邻居扩展（schema→evidence + same-session）。"""
from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.anchor_search import (
    PATH_SEMANTIC,
    AnchorNode,
)
from dual_mem.retrieval.graph_expander import GraphExpander
from dual_mem.types import Layer, MemoryNode, MemoryStatus


def _seed(factory, fake_embed, content: str, layer: Layer, *,
          session_id: str = "", node_id: str | None = None) -> MemoryNode:
    n = MemoryNode(
        content=content,
        layer=layer,
        app_id="app",
        user_id="u",
        session_id=session_id,
        status=MemoryStatus.ACTIVE,
    )
    if node_id:
        n.node_id = node_id
    n.embedding = fake_embed.embed_sync(content)
    factory.vector.upsert([n])
    return n


def test_expander_empty_anchors_returns_empty(tmp_storage, fake_embed):
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed, llm=None,
    )
    expander = GraphExpander(factory=factory)
    out = expander.expand(anchors=[], app_ids=["app"], user_id="u")
    assert out.expanded == []


def test_expander_session_neighbours(tmp_storage, fake_embed):
    """同 session 的 fact siblings 应被加为 timeline 邻居。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed, llm=None,
    )
    seed = _seed(factory, fake_embed, "用户喜欢咖啡", Layer.L4_IDENTITY, session_id="s1", node_id="seed")
    _seed(factory, fake_embed, "用户也爱旅行", Layer.L4_IDENTITY, session_id="s1", node_id="sib")
    _seed(factory, fake_embed, "本次会话摘要", Layer.L3_SUMMARY, session_id="s1", node_id="sum")
    _seed(factory, fake_embed, "另一个会话内容", Layer.L4_IDENTITY,
          session_id="s2", node_id="other")

    anchors = [AnchorNode(node=seed, score=0.8, source_path=PATH_SEMANTIC)]
    expander = GraphExpander(factory=factory)
    result = expander.expand(anchors=anchors, app_ids=["app"], user_id="u")

    expanded_ids = {a.node_id for a in result.expanded}
    assert "sib" in expanded_ids       # same session → timeline
    assert "sum" in expanded_ids       # same session → session summary
    assert "other" not in expanded_ids  # 不同 session 不应被引入
    # 邻居分数应低于 seed (attenuation)
    for anchor in result.expanded:
        assert anchor.score < 0.8


def test_expander_skips_non_active(tmp_storage, fake_embed):
    """SHADOW 状态的兄弟节点不应被加为邻居。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed, llm=None,
    )
    seed = _seed(factory, fake_embed, "种子", Layer.L4_IDENTITY, session_id="s1", node_id="seed")
    sib = _seed(factory, fake_embed, "兄弟", Layer.L4_IDENTITY, session_id="s1", node_id="sib")
    sib.status = MemoryStatus.SHADOW
    factory.vector.upsert([sib])

    anchors = [AnchorNode(node=seed, score=0.7, source_path=PATH_SEMANTIC)]
    expander = GraphExpander(factory=factory)
    result = expander.expand(anchors=anchors, app_ids=["app"], user_id="u")
    expanded_ids = {a.node_id for a in result.expanded}
    assert "sib" not in expanded_ids


def test_expander_no_session_no_expand(tmp_storage, fake_embed):
    """没有 session_id 的种子不会扩展（避免全用户广扩）。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed, llm=None,
    )
    seed = _seed(factory, fake_embed, "种子", Layer.L4_IDENTITY, session_id="", node_id="seed")
    _seed(factory, fake_embed, "其他", Layer.L4_IDENTITY, session_id="", node_id="other")

    anchors = [AnchorNode(node=seed, score=0.7, source_path=PATH_SEMANTIC)]
    expander = GraphExpander(factory=factory)
    result = expander.expand(anchors=anchors, app_ids=["app"], user_id="u")
    assert result.expanded == []
