import asyncio
import httpx
import pytest
from dual_mem.providers.jev import JevClient


def reply(request):
    import json

    body = json.loads(request.content)
    assert body["model"] == "jev-1.13.0"
    assert request.headers["Authorization"] == "Bearer test"
    return httpx.Response(
        200,
        json={
            "answers": {
                key: {
                    "type": "choice",
                    "choice": "YES",
                    "confidence": 0.9,
                    "probabilities": {"YES": 0.9, "NO": 0.1},
                }
                for key in body["questions"]
            },
            "usage": {"input_tokens": 12},
        },
    )


def questions(n):
    return {
        str(i): {
            "type": "choice",
            "instructions": "Decide",
            "criteria": {"YES": "yes", "NO": "no"},
        }
        for i in range(n)
    }


async def test_split_batches_and_injected_transport_authentication():
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as transport:
        jev = JevClient(
            endpoint="https://test/systemone",
            api_key="test",
            max_questions=2,
            http_client=transport,
        )
        answers = await jev.evaluate(state={"content": "中文"}, questions=questions(5))
        assert len(answers) == 5
        assert jev.request_count == 3 and jev.input_tokens == 36
        await jev.aclose()
        assert not transport.is_closed


async def test_client_concurrency_is_global_across_multiple_evaluations():
    active = peak = 0

    async def handler(req):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return reply(req)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        jev = JevClient(
            endpoint="https://test",
            api_key="test",
            concurrency=3,
            max_questions=1,
            http_client=transport,
        )
        await asyncio.gather(
            *(jev.evaluate(state={}, questions=questions(4)) for _ in range(3))
        )
        assert peak == 3
        assert jev.request_count == 12


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"answers": {}},
        {
            "answers": {
                "0": {
                    "type": "choice",
                    "choice": "UNKNOWN",
                    "confidence": 1,
                    "probabilities": {"UNKNOWN": 1},
                }
            }
        },
        {"answers": {"0": {"type": "noul", "noul": 0.5}}},
    ],
)
async def test_invalid_response_never_becomes_a_write_decision(payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    ) as transport:
        jev = JevClient(endpoint="https://test", api_key="test", http_client=transport)
        with pytest.raises(ValueError):
            await jev.evaluate(state={}, questions=questions(1))
        assert jev.request_count == 1


async def test_transient_retry_and_context_rejection():
    calls = 0
    events = []

    def handler(req):
        nonlocal calls
        calls += 1
        return httpx.Response(503) if calls == 1 else reply(req)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        jev = JevClient(
            endpoint="https://test",
            api_key="test",
            http_client=transport,
            usage_callback=events.append,
        )
        await jev.evaluate(state={}, questions=questions(1))
        assert calls == 2
        assert len(events) == 2
        assert [e.success for e in events] == [False, True]
        assert [e.status_code for e in events] == [503, 200]
        assert [e.attempt for e in events] == [1, 2]
        assert events[1].prompt_tokens == 12
        assert all(e.batch_size == 1 and e.latency_ms >= 0 for e in events)
        jev.max_question_chars = 100
        with pytest.raises(ValueError, match="context budget"):
            await jev.evaluate(state={"long": "x" * 200}, questions=questions(1))
        assert calls == 2


@pytest.mark.parametrize(
    "option",
    ["concurrency", "max_questions", "max_request_chars", "retry_attempts", "timeout"],
)
def test_invalid_limits(option):
    with pytest.raises(ValueError, match="positive"):
        JevClient(endpoint="https://test", api_key="test", **{option: 0})
