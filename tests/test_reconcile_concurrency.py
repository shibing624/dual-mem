"""ReconcilerWorker.reconcile_pending bounded-concurrency drain.

Verifies that reconcile_concurrency>1 still drains every queued task exactly once
(same result as serial), and that fast-write originals whose content a reconcile ADD
re-emits get shadowed (covered → replaced). The scripted reconcile echoes each new
memory's content so coverage matches.
"""
import pytest

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.system2.reconciler_worker import ReconcilerWorker
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from tests.conftest import FakeLLMClient


def _make_factory(tmp_storage, fake_embed, concurrency: int):
    settings = Settings(
        mode="system1",
        storage_dir=tmp_storage,
        reconcile_concurrency=concurrency,
    )
    # reconcile op: one ADD echoing the new memory content verbatim (no supersede), so the
    # worker's coverage check shadows the fast-write original it re-emitted.
    def _echo(*, system, user):
        content = user.split(". ", 1)[-1].split("\n", 1)[0].strip()
        return [{"op": "ADD", "content": content, "layer": "L2_FACT"}]

    llm = FakeLLMClient(responses={"reconcile": _echo})
    return ComponentFactory(settings=settings, embed=fake_embed, llm=llm)


def _seed_fast_write(factory, node_id: str, content: str) -> None:
    n = MemoryNode(
        content=content, layer=Layer.L2_FACT, app_id="app", user_id="u",
        status=MemoryStatus.ACTIVE, is_latest=True, node_id=node_id,
    )
    n.embedding = factory.embed.embed_sync(content)
    factory.vector.upsert([n])


def _enqueue_tasks(factory, n: int) -> None:
    for i in range(n):
        nid = f"fw_{i}"
        _seed_fast_write(factory, nid, f"fact number {i}")
        factory.cache.enqueue_reconcile_task(
            app_id="app", user_id="u", agent_id="", node_ids=[nid]
        )


@pytest.mark.parametrize("concurrency", [1, 8])
async def test_reconcile_pending_drains_all_tasks(tmp_storage, fake_embed, concurrency):
    factory = _make_factory(tmp_storage, fake_embed, concurrency)
    _enqueue_tasks(factory, 6)

    worker = ReconcilerWorker(factory=factory)
    processed = await worker.reconcile_pending(app_id="app", user_id="u", agent_id="")

    assert processed == 6
    # queue fully drained
    assert factory.cache.list_pending_reconcile_tasks(app_id="app", user_id="u") == []
    # each fast-write original was re-emitted by a reconcile ADD → covered → shadowed
    for i in range(6):
        node = factory.vector.get(f"fw_{i}")
        assert node.status is MemoryStatus.SHADOW
        assert node.is_latest is False


async def test_reconcile_concurrency_matches_serial_result(tmp_path, fake_embed):
    s_dir = tmp_path / "serial"
    p_dir = tmp_path / "parallel"
    s_dir.mkdir(parents=True, exist_ok=True)
    p_dir.mkdir(parents=True, exist_ok=True)

    serial = _make_factory(str(s_dir), fake_embed, 1)
    _enqueue_tasks(serial, 5)
    n_serial = await ReconcilerWorker(factory=serial).reconcile_pending(
        app_id="app", user_id="u", agent_id=""
    )

    parallel = _make_factory(str(p_dir), fake_embed, 8)
    _enqueue_tasks(parallel, 5)
    n_parallel = await ReconcilerWorker(factory=parallel).reconcile_pending(
        app_id="app", user_id="u", agent_id=""
    )

    assert n_serial == n_parallel == 5


async def test_failed_reconcile_task_remains_pending(
    tmp_storage,
    fake_embed,
    monkeypatch,
):
    factory = _make_factory(tmp_storage, fake_embed, 1)
    _enqueue_tasks(factory, 1)
    worker = ReconcilerWorker(factory=factory)

    async def _fail(_task):
        raise RuntimeError("reconcile failed")

    monkeypatch.setattr(worker, "_process_task", _fail)
    with pytest.raises(RuntimeError, match="reconcile failed"):
        await worker.reconcile_pending(app_id="app", user_id="u", agent_id="")

    tasks = factory.cache.list_pending_reconcile_tasks(app_id="app", user_id="u")
    assert len(tasks) == 1
