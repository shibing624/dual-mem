from dual_mem.system2.clustering import cluster_facts
from dual_mem.types import Layer, MemoryNode


def _vec(comps: dict[int, float], dim: int = 16) -> list[float]:
    v = [0.0] * dim
    for i, val in comps.items():
        v[i] = val
    return v


def _fact(nid: str, content: str, embedding: list[float]) -> MemoryNode:
    node = MemoryNode(
        content=content,
        layer=Layer.L2_FACT,
        app_id="app",
        user_id="u",
        node_id=nid,
    )
    node.embedding = embedding
    return node


# 簇内成员两两 cosine = 1/1.25 = 0.8（>=0.55 可聚类，<0.92 不去重）；
# 不同簇之间正交（cosine=0）。
def _cluster_a() -> list[MemoryNode]:
    return [
        _fact("a1", "用户做饭严格按菜谱", _vec({0: 1.0, 4: 0.5})),
        _fact("a2", "用户烘焙精确称量", _vec({0: 1.0, 5: 0.5})),
        _fact("a3", "用户煮咖啡按固定比例", _vec({0: 1.0, 6: 0.5})),
    ]


def _cluster_b() -> list[MemoryNode]:
    return [
        _fact("b1", "用户玩游戏追求全成就", _vec({1: 1.0, 7: 0.5})),
        _fact("b2", "用户看攻略通关", _vec({1: 1.0, 8: 0.5})),
        _fact("b3", "用户收集所有道具", _vec({1: 1.0, 9: 0.5})),
    ]


def test_two_clusters_found():
    facts = _cluster_a() + _cluster_b()
    clusters = cluster_facts(facts)
    assert len(clusters) == 2
    id_sets = {frozenset(c["ids"]) for c in clusters}
    assert frozenset({"a1", "a2", "a3"}) in id_sets
    assert frozenset({"b1", "b2", "b3"}) in id_sets
    for c in clusters:
        assert len(c["centroid_embedding"]) == 16
        assert c["centroid_text"]
        assert all(f["layer"] == "L2_FACT" for f in c["facts"])


def test_too_few_facts_no_cluster():
    facts = _cluster_a()[:2]
    assert cluster_facts(facts) == []


def test_noise_points_excluded():
    facts = _cluster_a() + [
        _fact("n1", "孤立事实一", _vec({2: 1.0, 10: 0.5})),
        _fact("n2", "孤立事实二", _vec({3: 1.0, 11: 0.5})),
    ]
    clusters = cluster_facts(facts)
    assert len(clusters) == 1
    assert frozenset(clusters[0]["ids"]) == frozenset({"a1", "a2", "a3"})


def test_intra_cluster_dedup():
    facts = _cluster_a() + [_fact("a1dup", "用户做饭严格按菜谱", _vec({0: 1.0, 4: 0.5}))]
    clusters = cluster_facts(facts)
    assert len(clusters) == 1
    # a1dup 与 a1 cosine=1.0 >= 0.92 被去重，只保留 3 条代表
    assert len(clusters[0]["ids"]) == 3
