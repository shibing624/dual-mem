"""non_destructive guard: reconcile must NOT shadow fast-write originals.

Regression test for the `_shadow_covered_originals` short-circuit in
ReconcilerWorker._process_task. When reconcile_non_destructive=True the contract is
"原始 fact 只增不减": even if the reconcile LLM re-emits an original's content verbatim
as an ADD/SUPPLEMENT (content coverage matches), the original fast-write node must stay
ACTIVE / is_latest=True. The destructive (default) path is the control: there the same
re-emit DOES shadow the original.

Pairs with test_reconcile_concurrency.py, which asserts the destructive baseline.
"""
import pytest

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.system2.reconciler_worker import ReconcilerWorker
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


def _make_factory(tmp_storage, fake_embed, *, non_destructive: bool):
    settings = Settings(
        mode="system1",
        storage_dir=tmp_storage,
        reconcile_concurrency=1,
        reconcile_non_destructive=non_destructive,
        # keep the LLM in the loop: we want to exercise the
        # non_destructive + skip_llm=False residual path, not the skip_llm short-circuit.
        reconcile_skip_llm=False,
        # force the weak-candidate fast path off so the reconcile LLM actually runs.
        reconcile_weak_candidate_score=0.0,
    )

    # reconcile op: one ADD echoing the new memory content verbatim (no supersede). This
    # makes the worker's content-coverage check match the fast-write original, which under
    # the destructive path triggers _shadow_covered_originals.
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


async def test_non_destructive_keeps_originals_active(tmp_storage, fake_embed):
    """non_destructive=True: re-emitted originals stay ACTIVE (never shadowed)."""
    factory = _make_factory(tmp_storage, fake_embed, non_destructive=True)
    _enqueue_tasks(factory, 5)

    processed = await ReconcilerWorker(factory=factory).reconcile_pending(
        app_id="app", user_id="u", agent_id=""
    )

    assert processed == 5
    for i in range(5):
        node = factory.vector.get(f"fw_{i}")
        assert node is not None
        assert node.status is MemoryStatus.ACTIVE, (
            f"fw_{i} was shadowed under non_destructive — immutable guarantee broken"
        )
        assert node.is_latest is True


async def test_destructive_baseline_shadows_originals(tmp_storage, fake_embed):
    """Control: non_destructive=False, the same re-emit DOES shadow the original.

    Confirms the shadow logic itself is intact, so the test above proves the guard
    rather than a globally broken coverage check.
    """
    factory = _make_factory(tmp_storage, fake_embed, non_destructive=False)
    _enqueue_tasks(factory, 5)

    processed = await ReconcilerWorker(factory=factory).reconcile_pending(
        app_id="app", user_id="u", agent_id=""
    )

    assert processed == 5
    for i in range(5):
        node = factory.vector.get(f"fw_{i}")
        assert node is not None
        assert node.status is MemoryStatus.SHADOW
        assert node.is_latest is False