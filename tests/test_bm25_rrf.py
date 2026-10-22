from dual_mem.retrieval.bm25 import (
    compute_bm25_confidences, compute_bm25_scores, score_and_rank, tokenize,
)
from dual_mem.retrieval.rrf import rrf_fuse


def test_tokenize_mixed():
    assert tokenize("Hello 世界 World") == ["hello", "世界", "world"]


def test_bm25_more_query_terms_ranks_first():
    candidates = [
        ("a", "python machine learning tutorial"),
        ("b", "cooking recipe for dinner"),
        ("c", "python basics"),
    ]
    ranked = score_and_rank(["python", "machine", "learning"], candidates)
    assert ranked[0][0] == "a"
    assert ranked[0][1] == 1.0


def test_bm25_empty_query_all_zero():
    scores = compute_bm25_scores([], ["foo bar", "baz"])
    assert scores == [0.0, 0.0]


def test_bm25_confidence_penalizes_ubiquitous_partial_match():
    terms = tokenize("project status quasar-917")
    contents = [
        "project status meeting notes",
        "project status remains unchanged",
        "project status for quasar-917",
    ]
    raw = compute_bm25_scores(terms, contents)
    confidence = compute_bm25_confidences(terms, contents, raw)
    assert confidence[2] > 0.5
    assert 0.0 < confidence[0] < 0.2
    assert 0.0 < confidence[1] < 0.2


def test_rrf_multi_channel_doc_ranks_first():
    vec = [{"node_id": "x", "node": "X"}, {"node_id": "y", "node": "Y"}]
    bm25 = [{"node_id": "x", "node": "X"}, {"node_id": "z", "node": "Z"}]
    fused = rrf_fuse({"vec": vec, "bm25": bm25})
    assert fused[0]["node_id"] == "x"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_rrf_weights_scale_channel():
    vec = [{"node_id": "a"}]
    bm25 = [{"node_id": "b"}]
    fused = rrf_fuse({"vec": vec, "bm25": bm25}, weights={"vec": 2.0, "bm25": 0.5})
    by_id = {f["node_id"]: f["rrf_score"] for f in fused}
    assert by_id["a"] > by_id["b"]
