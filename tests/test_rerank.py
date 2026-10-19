import httpx
import pytest

from dual_mem.providers.rerank import RerankClient
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode
from conftest import FakeLLMClient
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


async def test_rerank_scores_are_restored_to_input_order():
    def respond(request):
        return httpx.Response(200, json={"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = RerankClient(endpoint="https://example.test/rerank", http_client=http)
        assert await client.rerank(query="q", documents=["a", "b"]) == [0.1, 0.9]
        assert await client.rerank(query="q", documents=[]) == []


@pytest.mark.parametrize("rows", [
    [{"index": 0, "relevance_score": 0.9}],
    [{"index": 0, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}],
    [{"index": 0, "relevance_score": 0.9}, {"index": 2, "relevance_score": 0.1}],
    [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 1.5}],
])
async def test_rerank_rejects_missing_duplicate_or_invalid_scores(rows):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"results": rows})
    )) as http:
        client = RerankClient(endpoint="https://example.test/rerank", http_client=http)
        with pytest.raises(ValueError):
            await client.rerank(query="q", documents=["a", "b"])


async def test_rerank_http_failure_is_not_retried():
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(503)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = RerankClient(endpoint="https://example.test/rerank", http_client=http)
        with pytest.raises(httpx.HTTPStatusError):
            await client.rerank(query="q", documents=["a"])
    assert len(calls) == 1


async def test_cross_encoder_selects_from_candidate_pool_before_top_k(tmp_storage, fake_embed):
    class Reranker:
        async def rerank(self, *, query, documents):
            return [0.9 if "authorization" in text else 0.1 for text in documents]

    client = MemoryClient(settings=Settings(storage_dir=tmp_storage), embed=fake_embed,
                          llm=FakeLLMClient())
    client.factory._reranker = Reranker()
    client.factory.vector.upsert([
        MemoryNode(node_id="answer", content="return authorization RMA-81723",
                   layer=Layer.L1_FACT, app_id="app", user_id="u",
                   embedding=fake_embed.embed_sync("return authorization RMA-81723")),
        MemoryNode(node_id="noise", content="holiday plans",
                   layer=Layer.L1_FACT, app_id="app", user_id="u",
                   embedding=fake_embed.embed_sync("holiday plans")),
    ])
    result = await client.search(query="how do I return this?", app_ids=["app"], user_id="u",
                                 limit=1, min_score=0)
    assert result.memories.normal[0].memory_id == "answer"
    assert result.memories.normal[0].score == 0.9
    await client.aclose()
