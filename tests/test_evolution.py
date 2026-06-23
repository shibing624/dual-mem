from dual_mem.retrieval.evolution import expand_evolution_chains
from dual_mem.types import Layer, MemoryNode


class FakeVector:
    def __init__(self, nodes: list[MemoryNode]):
        self._by_id = {n.node_id: n for n in nodes}

    def get(self, node_id: str) -> MemoryNode | None:
        return self._by_id.get(node_id)

    def get_by_ids(self, node_ids: list[str]) -> dict[str, MemoryNode]:
        return {nid: self._by_id[nid] for nid in node_ids if nid in self._by_id}


def _node(node_id, gmt, *, is_latest, supersedes=(), superseded_by=(), content=""):
    return MemoryNode(
        content=content or node_id,
        layer=Layer.L2_FACT,
        app_id="app",
        user_id="u",
        node_id=node_id,
        gmt_created=gmt,
        is_latest=is_latest,
        supersedes=list(supersedes),
        superseded_by=list(superseded_by),
    )


def test_chain_hit_returns_head_with_full_chain():
    # A(newest, latest) -> B -> C(oldest)
    a = _node("A", 300, is_latest=True, supersedes=["B"])
    b = _node("B", 200, is_latest=False, supersedes=["C"], superseded_by=["A"])
    c = _node("C", 100, is_latest=False, superseded_by=["B"])
    vector = FakeVector([a, b, c])

    hit_b = _node("B", 200, is_latest=False, supersedes=["C"], superseded_by=["A"])
    hit_b.score = 0.8
    result = expand_evolution_chains(vector=vector, hits=[hit_b])

    assert len(result) == 1
    item = result[0]
    assert item["is_evolved"] is True
    assert item["node"].node_id == "A"
    assert item["score"] == 0.8
    chain_ids = [c["node_id"] for c in item["evolution_chain"]]
    assert chain_ids == ["A", "B", "C"]
    assert item["evolution_chain"][0]["layer"] == "L2_FACT"


def test_plain_node_passes_through():
    plain = _node("X", 100, is_latest=True)
    plain.score = 0.5
    result = expand_evolution_chains(vector=FakeVector([plain]), hits=[plain])

    assert len(result) == 1
    assert result[0]["is_evolved"] is False
    assert result[0]["evolution_chain"] is None
    assert result[0]["node"].node_id == "X"
    assert result[0]["score"] == 0.5


def test_same_chain_multiple_hits_dedup_keep_highest_score():
    a = _node("A", 300, is_latest=True, supersedes=["B"])
    b = _node("B", 200, is_latest=False, supersedes=["C"], superseded_by=["A"])
    c = _node("C", 100, is_latest=False, superseded_by=["B"])
    vector = FakeVector([a, b, c])

    hit_b = _node("B", 200, is_latest=False, supersedes=["C"], superseded_by=["A"])
    hit_b.score = 0.6
    hit_c = _node("C", 100, is_latest=False, superseded_by=["B"])
    hit_c.score = 0.9
    result = expand_evolution_chains(vector=vector, hits=[hit_b, hit_c])

    assert len(result) == 1
    assert result[0]["node"].node_id == "A"
    assert result[0]["score"] == 0.9
