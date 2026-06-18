import httpx
import respx

from dual_mem.providers.embedding import EmbedService


def _make_service(dim=4):
    return EmbedService(
        base_url="https://api.test/v1", api_key="sk-x", model="embed-test", dim=dim
    )


def _embed_response(vectors: list[list[float]]) -> dict:
    return {
        "object": "list",
        "model": "embed-test",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


@respx.mock
def test_embed_single():
    respx.post("https://api.test/v1/embeddings").mock(
        return_value=httpx.Response(200, json=_embed_response([[0.1, 0.2, 0.3, 0.4]]))
    )
    vec = _make_service().embed("hello")
    assert vec == [0.1, 0.2, 0.3, 0.4]


@respx.mock
def test_embed_batch():
    respx.post("https://api.test/v1/embeddings").mock(
        return_value=httpx.Response(
            200, json=_embed_response([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        )
    )
    vecs = _make_service().embed_batch(["a", "b"])
    assert vecs == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]


def test_embed_batch_empty_returns_empty():
    assert _make_service().embed_batch([]) == []
