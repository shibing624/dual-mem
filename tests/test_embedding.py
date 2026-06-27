import json

import httpx
import respx

from dual_mem.config import CHARS_PER_TOKEN, EMBED_MAX_TOKENS
from dual_mem.providers.embedding import EmbedService
from dual_mem.providers.usage import UsageEvent


def _make_service(dim: int = 4, model: str = "embed-test") -> EmbedService:
    return EmbedService(
        base_url="https://api.test/v1",
        api_key="sk-x",
        model=model,
        dim=dim,
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


@respx.mock
async def test_embed_batch_never_sends_dimensions_param():
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
async def test_embed_batch_bge_m3_omits_dimensions():
    captured: dict = {}

    def _record_request(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_embed_response([[0.1] * 4]))

    respx.post("https://api.test/v1/embeddings").mock(side_effect=_record_request)
    await _make_service(dim=1024, model="embed").embed("hello")
    assert "dimensions" not in captured["json"]


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


@respx.mock
async def test_embed_batch_long_text_head_truncated():
    max_chars = int(EMBED_MAX_TOKENS * CHARS_PER_TOKEN)
    long_text = "x" * (max_chars + 2000)
    captured: dict = {}

    def _record_request(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        n = len(captured["json"]["input"])
        return httpx.Response(
            200,
            json=_embed_response([[float(i + 1)] * 4 for i in range(n)]),
        )

    respx.post("https://api.test/v1/embeddings").mock(side_effect=_record_request)
    svc = EmbedService(
        base_url="https://api.test/v1",
        api_key="sk-x",
        model="embed-test",
        dim=4,
        input_max_tokens=EMBED_MAX_TOKENS,
        chars_per_token=CHARS_PER_TOKEN,
    )
    vecs = await svc.embed_batch([long_text, "hi"])
    # Head-truncated, NOT chunked: exactly one input per text, no extra chunk rows.
    sent = captured["json"]["input"]
    assert len(sent) == 2
    assert len(sent[0]) == max_chars
    assert sent[1] == "hi"
    assert len(vecs) == 2
    assert len(vecs[0]) == 4


def test_embed_max_tokens_derived_char_budget():
    svc = EmbedService(
        base_url="https://api.test/v1",
        api_key="sk-x",
        model="embed-test",
        input_max_tokens=8192,
        chars_per_token=2.5,
    )
    assert svc.input_max_chars == 20480
    assert svc.input_max_tokens == 8192
