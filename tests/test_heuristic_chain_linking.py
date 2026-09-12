"""Unit tests for ``link_evolution_chains_heuristic`` — the zero-LLM chain linker.

NOTE (scope, 2026-09): this heuristic is **no longer wired into the digest pipeline**.
It used to back-fill ``supersedes``/``superseded_by`` pointers on the ``reconcile_skip_llm``
path by grouping ACTIVE L2/L4 nodes that share a tag. Measured on the real workload it
chained ~62% of all memories, and because retrieval expanded every chain into the QA
context, superseded intermediate states were fed to the model as if they were still
current — which cost accuracy (PersonaMem 4/4 affected questions flipped correct → wrong).
``System2Writer._digest_user`` therefore drains the reconcile queue without linking, and
the ``reconcile_link_chains_heuristic`` setting is currently inert.

The function is kept and unit-tested here as a standalone utility (it can be called
explicitly, e.g. to reconstruct a preference timeline offline). These tests cover the
function directly rather than asserting pipeline behaviour that no longer exists.
"""
from conftest import FakeLLMClient

from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.system2.reconciler_worker import link_evolution_chains_heuristic
from dual_mem.types import Layer, MemoryNode


def test_heuristic_disabled_leaves_nodes_unlinked(tmp_storage, fake_embed):
    """With the heuristic off, pre-existing nodes keep their empty chain pointers."""
    client = MemoryClient(
        settings=Settings(
            mode="dual",
            storage_dir=tmp_storage,
            reconcile_sync=False,
            reconcile_skip_llm=True,
            reconcile_link_chains_heuristic=False,
        ),
        embed=fake_embed,
        llm=FakeLLMClient(),
    )

    vector = client.factory.vector
    vector.upsert(
        [
            MemoryNode(
                node_id="old",
                app_id="app",
                user_id="u",
                layer=Layer.L4_IDENTITY,
                content="用户最喜欢的编程语言是 Java",
                embedding=[1.0, 0.0],
                tags=["lang"],
                memory_at=1_600_000_000,
            ),
            MemoryNode(
                node_id="new",
                app_id="app",
                user_id="u",
                layer=Layer.L4_IDENTITY,
                content="用户现在主要使用 Python",
                embedding=[1.0, 0.0],
                tags=["lang"],
                memory_at=1_700_000_000,
            ),
        ]
    )

    # Disabled via settings: the digest path must not wire anything.
    assert client.settings.reconcile_link_chains_heuristic is False
    nodes = vector.get_many({"$and": [{"app_id": "app"}, {"user_id": "u"}]}, limit=50)
    assert all(not (n.supersedes or []) for n in nodes)
    assert all(not (n.superseded_by or []) for n in nodes)


def test_heuristic_links_same_tag_nodes_and_keeps_them_active(tmp_storage, fake_embed):
    """Same layer + shared tag + ordered by time → newer supersedes older, both ACTIVE.

    "Link but don't hide": the old node keeps ``status=ACTIVE`` / ``is_latest=True`` so the
    timeline can be reconstructed without sacrificing recall.
    """
    client = MemoryClient(
        settings=Settings(mode="dual", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(),
    )
    vector = client.factory.vector
    vector.upsert(
        [
            MemoryNode(
                node_id="old",
                app_id="app",
                user_id="u",
                layer=Layer.L4_IDENTITY,
                content="用户最喜欢的编程语言是 Java",
                embedding=[1.0, 0.0],
                tags=["lang"],
                memory_at=1_600_000_000,
            ),
            MemoryNode(
                node_id="new",
                app_id="app",
                user_id="u",
                layer=Layer.L4_IDENTITY,
                content="用户现在主要使用 Python",
                embedding=[1.0, 0.0],
                tags=["lang"],
                memory_at=1_700_000_000,
            ),
        ]
    )

    linked = link_evolution_chains_heuristic(client.factory, app_id="app", user_id="u")
    assert linked >= 1

    old = vector.get("old")
    new = vector.get("new")
    assert old.node_id in (new.supersedes or [])
    assert new.node_id in (old.superseded_by or [])
    # keep_active: the older version stays recallable, it is not hidden.
    assert old.status.value == "ACTIVE"
    assert old.is_latest is True


def test_heuristic_does_not_link_across_layers_or_blank_tags(
    tmp_storage,
    fake_embed,
):
    """Only same-layer + non-blank shared-tag nodes are eligible for linking."""
    client = MemoryClient(
        settings=Settings(
            mode="dual",
            storage_dir=tmp_storage,
        ),
        embed=fake_embed,
        llm=FakeLLMClient(),
    )
    vector = client.factory.vector
    vector.upsert(
        [
            MemoryNode(
                node_id="fact",
                app_id="app",
                user_id="u",
                layer=Layer.L2_FACT,
                content="fact",
                embedding=[1.0, 0.0],
                tags=["lang"],
            ),
            MemoryNode(
                node_id="identity",
                app_id="app",
                user_id="u",
                layer=Layer.L4_IDENTITY,
                content="identity",
                embedding=[1.0, 0.0],
                tags=["lang"],
            ),
            MemoryNode(
                node_id="blank-old",
                app_id="app",
                user_id="u",
                layer=Layer.L4_IDENTITY,
                content="blank old",
                embedding=[1.0, 0.0],
                tags=[" "],
            ),
            MemoryNode(
                node_id="blank-new",
                app_id="app",
                user_id="u",
                layer=Layer.L4_IDENTITY,
                content="blank new",
                embedding=[1.0, 0.0],
                tags=[""],
            ),
        ]
    )

    linked = link_evolution_chains_heuristic(
        client.factory,
        app_id="app",
        user_id="u",
    )

    assert linked == 0
