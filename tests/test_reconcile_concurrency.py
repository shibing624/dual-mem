import pytest
from conftest import FakeJev, FakeLLMClient
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.system2.reconciler_worker import ReconcilerWorker
from dual_mem.types import MemoryNode, Layer


def setup(tmp_storage, embed):
    c = MemoryClient(
        settings=Settings(mode="dual", storage_dir=tmp_storage),
        embed=embed,
        llm=FakeLLMClient(),
        jev=FakeJev(),
    )
    n = MemoryNode(
        node_id="legacy",
        content="old extracted fact",
        layer=Layer.L1_FACT,
        app_id="app",
        user_id="u",
        embedding=embed.embed_sync("old"),
    )
    c.factory.vector.upsert([n])
    c.factory.cache.enqueue_reconcile_task(
        app_id="app", user_id="u", agent_id="", node_ids=["legacy"]
    )
    return c


async def test_legacy_pending_task_is_decided_and_archived_without_deletion(
    tmp_storage, fake_embed
):
    c = setup(tmp_storage, fake_embed)
    assert (
        await ReconcilerWorker(factory=c.factory).reconcile_pending(
            app_id="app", user_id="u"
        )
        == 1
    )
    assert not c.factory.vector.get("legacy").is_latest
    assert c.factory.vector.get("legacy").content == "old extracted fact"
    assert c.factory.cache.reconcile_queue_size() == 0
    assert len(c.factory.jev.calls) == 1 and c.factory.cache.list_pending_s2_scopes()
    await c.aclose()


async def test_legacy_failure_remains_pending(tmp_storage, fake_embed):
    c = setup(tmp_storage, fake_embed)
    c.factory.jev.responses = RuntimeError("failed")
    with pytest.raises(RuntimeError):
        await ReconcilerWorker(factory=c.factory).reconcile_pending(
            app_id="app", user_id="u"
        )
    assert (
        c.factory.cache.reconcile_queue_size() == 1
        and c.factory.vector.get("legacy").is_latest
    )
    await c.aclose()


async def test_legacy_missing_source_never_acknowledged(tmp_storage, fake_embed):
    c = setup(tmp_storage, fake_embed)
    c.factory.vector.delete(["legacy"])
    with pytest.raises(RuntimeError, match="missing evidence"):
        await ReconcilerWorker(factory=c.factory).reconcile_pending(
            app_id="app", user_id="u"
        )
    assert c.factory.cache.reconcile_queue_size() == 1
    await c.aclose()
