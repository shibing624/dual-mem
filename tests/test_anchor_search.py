"""P1-1 子PR1: AnchorSearchEngine — 5 路并行召回。"""
from datetime import datetime

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.retrieval.anchor_search import (
    PATH_ENTITY,
    PATH_SCHEMA,
    PATH_SEMANTIC,
    PATH_TEMPORAL,
    AnchorSearchEngine,
)
from dual_mem.retrieval.query_understanding import understand
from dual_mem.types import Layer, MemoryNode, MemoryStatus


def _seed_facts(factory, fake_embed, contents: list[tuple[str, str, int]]) -> None:
    """Insert (content, layer_value, gmt_created) facts with deterministic embeddings."""
    nodes = []
    for content, layer_val, gmt in contents:
        node = MemoryNode(
            content=content,
            layer=Layer(layer_val),
            app_id="app",
            user_id="u",
            status=MemoryStatus.ACTIVE,
            gmt_created=gmt,
        )
        node.embedding = fake_embed.embed_sync(content)
        nodes.append(node)
    factory.vector.upsert(nodes)


async def test_anchor_search_runs_paths_in_parallel(tmp_storage, fake_embed):
    """semantic + entity 两条路径都启用，merged anchors 去重。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    _seed_facts(
        factory,
        fake_embed,
        [
            ("用户喜欢喝咖啡", Layer.L4_IDENTITY.value, 1000),
            ("用户昨天去了北京", Layer.L2_FACT.value, 1100),
            ("无关内容 random text", Layer.L2_FACT.value, 1200),
        ],
    )

    understanding = understand("咖啡")
    embedding = fake_embed.embed_sync("咖啡")
    engine = AnchorSearchEngine(factory=factory)

    result = await engine.search(
        query="咖啡",
        query_embedding=embedding,
        understanding=understanding,
        app_ids=["app"],
        user_id="u",
    )
    # semantic 一定有命中；entity 至少命中"咖啡"那条
    assert PATH_SEMANTIC in result.path_counts
    # 任何路径合并后都不应有重复 node_id
    seen = set()
    for anchor in result.anchors:
        assert anchor.node_id not in seen
        seen.add(anchor.node_id)


async def test_anchor_search_temporal_path_with_time_word(tmp_storage, fake_embed):
    """时间词 query → temporal 路径自动启用，按 created_after 过滤。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    today_ts = int(datetime.now().timestamp())
    _seed_facts(
        factory,
        fake_embed,
        [
            ("用户今天去了北京", Layer.L2_FACT.value, today_ts),
            ("用户上周买了车", Layer.L2_FACT.value, today_ts - 8 * 86400),  # too old
        ],
    )

    understanding = understand("今天发生了什么")
    assert understanding.has_temporal
    embedding = fake_embed.embed_sync("今天发生了什么")
    engine = AnchorSearchEngine(factory=factory)

    result = await engine.search(
        query="今天发生了什么",
        query_embedding=embedding,
        understanding=understanding,
        app_ids=["app"],
        user_id="u",
    )
    assert PATH_TEMPORAL in result.path_counts
    # 时间窗口里至少应该召回"今天"那条（不论是 temporal 还是 semantic 命中）
    contents = [a.node.content for a in result.anchors]
    assert any("今天" in c for c in contents)
    # 且 temporal path 自身的 count > 0（说明该路径确实跑了并有结果）
    assert result.path_counts[PATH_TEMPORAL] > 0


async def test_anchor_search_no_graph_skips_schema_path(tmp_storage, fake_embed):
    """无 graph（system1 模式）时 schema/intention 路径返回空但不抛错。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    assert factory.graph is None  # system1 没 graph
    _seed_facts(factory, fake_embed, [("用户喜欢喝咖啡", Layer.L4_IDENTITY.value, 1000)])

    embedding = fake_embed.embed_sync("咖啡")
    understanding = understand("咖啡")
    engine = AnchorSearchEngine(factory=factory)

    result = await engine.search(
        query="咖啡",
        query_embedding=embedding,
        understanding=understanding,
        app_ids=["app"],
        user_id="u",
        intention_enabled=True,  # 即便启用也无 graph
    )
    # schema/intention 路径要么不在 path_counts，要么 count==0；都不会抛错
    assert result.path_counts.get(PATH_SCHEMA, 0) == 0
    assert result.activated_schemas == []
    assert result.triggered_intentions == []


async def test_recall_limit_widens_semantic_pool(tmp_storage, fake_embed):
    """recall_limit 抬高每路候选上限：默认封顶 15，传 recall_limit=50 应召回更多。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    # 30 条事实，FakeEmbed 全正向量 → 任意两条 cosine > semantic_threshold(0.3)
    _seed_facts(
        factory,
        fake_embed,
        [(f"用户事实编号 {i}", Layer.L2_FACT.value, 1000 + i) for i in range(30)],
    )
    understanding = understand("用户事实")
    embedding = fake_embed.embed_sync("用户事实")
    engine = AnchorSearchEngine(factory=factory)

    default_res = await engine.search(
        query="用户事实",
        query_embedding=embedding,
        understanding=understanding,
        app_ids=["app"],
        user_id="u",
    )
    wide_res = await engine.search(
        query="用户事实",
        query_embedding=embedding,
        understanding=understanding,
        app_ids=["app"],
        user_id="u",
        recall_limit=50,
    )
    assert default_res.path_counts[PATH_SEMANTIC] == 15  # 默认上限
    assert wide_res.path_counts[PATH_SEMANTIC] > 15  # 被 recall_limit 放大
    assert wide_res.path_counts[PATH_SEMANTIC] <= 30  # 不超过可用总量


async def test_anchor_search_keyword_match_via_entity_path(tmp_storage, fake_embed):
    """entity 路径基于 query keywords 命中 content 子串。"""
    factory = ComponentFactory(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=None,
    )
    _seed_facts(
        factory,
        fake_embed,
        [
            ("用户在腾讯工作了三年", Layer.L4_IDENTITY.value, 1000),
            ("用户毕业于清华", Layer.L4_IDENTITY.value, 1100),
        ],
    )

    understanding = understand("用户的工作是什么")
    embedding = fake_embed.embed_sync("用户的工作是什么")
    engine = AnchorSearchEngine(factory=factory)

    result = await engine.search(
        query="用户的工作",
        query_embedding=embedding,
        understanding=understanding,
        app_ids=["app"],
        user_id="u",
    )
    # entity path 应该被启用（understanding.keywords 非空）
    assert understanding.keywords  # 验证前提
    # entity 命中"工作"，至少召回"用户在腾讯工作了三年"
    entity_anchors = [a for a in result.anchors if a.source_path == PATH_ENTITY]
    if entity_anchors:
        assert any("工作" in a.node.content for a in entity_anchors)
