import asyncio
import json
import pytest
from conftest import FakeLLMClient, FakeJev
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.system2.system2_agent import System2Agent


def make(tmp_storage, embed):
    def response(system, user):
        ids = [f["id"] for f in json.loads(user)["facts"]]
        return {"memories": [], "processed_ids": ids}

    return MemoryClient(
        settings=Settings(
            mode="dual", storage_dir=tmp_storage, purge_done_queues=False
        ),
        embed=embed,
        llm=FakeLLMClient({"json": response}),
        jev=FakeJev(),
    )


async def test_dual_only_cognizes_after_explicit_digest(tmp_storage, fake_embed):
    c = make(tmp_storage, fake_embed)
    await c.add(content="fact", user_id="u")
    assert (
        len(c.factory.llm.calls) == 1
        and len(c.factory.cache.list_pending_s2_scopes()) == 1
    )
    assert (await c.digest()).processed == 1
    assert not c.factory.cache.list_pending_s2_scopes()
    assert len(c.factory.llm.calls) == 2
    await c.aclose()


async def test_digest_watermark_preserves_new_arrivals(
    tmp_storage, fake_embed, monkeypatch
):
    c = make(tmp_storage, fake_embed)
    c.factory.cache.enqueue_s2_task("u", "default")

    async def run(self, **kwargs):
        c.factory.cache.enqueue_s2_task("u", "default")
        return {}

    monkeypatch.setattr(System2Agent, "run", run)
    assert (await c.digest()).processed == 1
    pending = c.factory.cache.list_pending_s2_scopes()
    assert len(pending) == 1 and pending[0]["id"] == 2
    await c.aclose()


async def test_partial_scope_failure_acks_only_successful_scopes(
    tmp_storage, fake_embed, monkeypatch
):
    c = make(tmp_storage, fake_embed)
    for uid in ("good", "bad"):
        c.factory.cache.enqueue_s2_task(uid, "default")

    async def run(self, **kwargs):
        if kwargs["user_id"] == "bad":
            raise RuntimeError("failed scope")
        return {}

    monkeypatch.setattr(System2Agent, "run", run)
    with pytest.raises(RuntimeError, match="failed scope"):
        await c.digest()
    assert [t["user_id"] for t in c.factory.cache.list_pending_s2_scopes()] == ["bad"]
    await c.aclose()


async def test_digest_scopes_use_configured_parallelism(
    tmp_storage, fake_embed, monkeypatch
):
    c = make(tmp_storage, fake_embed)
    c.settings.reconcile_concurrency = 3
    for i in range(8):
        c.factory.cache.enqueue_s2_task(str(i), "default")
    active = peak = 0

    async def run(self, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {}

    monkeypatch.setattr(System2Agent, "run", run)
    assert (await c.digest()).processed == 8 and peak == 3
    await c.aclose()


async def test_profile_only_update_queues_cognition(tmp_storage, fake_embed):
    c = make(tmp_storage, fake_embed)
    c.factory.llm.responses["extract"] = {"basic_info": {"location": "Beijing"}}
    c.factory.jev.responses = {"p0:representation": "PROFILE"}
    result = await c.add(content="Beijing", user_id="u")
    assert result.extracted_count == 1 and c.factory.cache.list_pending_s2_scopes()
    await c.aclose()
