import json

import httpx
import respx

from dual_mem.providers.embedding import EmbedService, embedding_api_dimensions
from dual_mem.providers.usage import UsageEvent


def _make_service(dim=4, model="embed-test"):
    return EmbedService(
        base_url="https://api.test/v1", api_key="sk-x", model=model, dim=dim
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
async def test_embed_single():
    respx.post("https://api.test/v1/embeddings").mock(
        return_value=httpx.Response(200, json=_embed_response([[0.1, 0.2, 0.3, 0.4]]))
    )
    vec = await _make_service().embed("hello")
    assert vec == [0.1, 0.2, 0.3, 0.4]


@respx.mock
async def test_embed_batch():
    respx.post("https://api.test/v1/embeddings").mock(
        return_value=httpx.Response(
            200, json=_embed_response([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        )
    )
    vecs = await _make_service().embed_batch(["a", "b"])
    assert vecs == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]


async def test_embed_batch_empty_returns_empty():
    assert await _make_service().embed_batch([]) == []


def test_embedding_api_dimensions_qwen3_omits_param():
    assert embedding_api_dimensions("qwen3-embedding-8b", 1536) is None
    assert embedding_api_dimensions("foo-for-online", 768) is None


def test_embedding_api_dimensions_openai_passes_dim():
    assert embedding_api_dimensions("text-embedding-3-small", 1536) == 1536


@respx.mock
async def test_embed_batch_qwen3_omits_dimensions_in_request():
    captured: dict = {}

    def _record_request(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200, json=_embed_response([[0.1, 0.2, 0.3, 0.4]])
        )

    respx.post("https://api.test/v1/embeddings").mock(side_effect=_record_request)
    await _make_service(dim=4096, model="qwen3-embedding-8b").embed("hello")
    assert "dimensions" not in captured["json"]
    assert captured["json"]["model"] == "qwen3-embedding-8b"


@respx.mock
async def test_embed_batch_openai_sends_dimensions():
    captured: dict = {}

    def _record_request(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200, json=_embed_response([[0.1, 0.2, 0.3, 0.4]])
        )

    respx.post("https://api.test/v1/embeddings").mock(side_effect=_record_request)
    await _make_service(dim=1536, model="text-embedding-3-small").embed("hello")
    assert captured["json"]["dimensions"] == 1536


@respx.mock
async def test_embed_batch_usage_callback():
    events: list[UsageEvent] = []

    def _cb(event: UsageEvent) -> None:
        events.append(event)

    respx.post("https://api.test/v1/embeddings").mock(
        return_value=httpx.Response(200, json=_embed_response([[0.1, 0.2, 0.3, 0.4]]))
    )
    svc = _make_service()
    svc.usage_callback = _cb
    await svc.embed_batch(["hello"])
    assert len(events) == 1
    assert events[0].kind == "embed_batch"
    assert events[0].text_chars == 5
    assert events[0].batch_size == 1
