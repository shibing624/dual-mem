"""Concurrency regression for the per-user write lock.

Same-user concurrent add() must serialize through the fast-write -> reconcile pipeline,
otherwise the evolution chain forks: several nodes that should have superseded one
another all end up ACTIVE and ``is_latest``, so the same business state is injected into
context more than once. Different users must NOT block each other (the write lock is
keyed per user).

The reconcile fake below deliberately forms a chain (each new memory supersedes the
current head), so "no fork" is a measurable invariant rather than a tautology:
with the lock the chain is linear (1 ACTIVE node), without it the writes race and fork.
"""
import asyncio
import re

from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.isolation import build_filter
from dual_mem.types import Layer, MemoryStatus

from conftest import FakeLLMClient


def _identity_response(seq: int) -> dict:
    return {
        "is_ephemeral": False,
        "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
        "identity": [
            {"content": f"用户喜欢喝咖啡{seq}", "speculate": None, "tags": ["food"]}
        ],
        "facts": [],
        "intentions": [],
        "basic_info": {},
    }


def _reconcile_supersede_head(*, system: str, user: str) -> list:
    """Supersede the first existing memory with the new one, forming a chain.

    ``system`` holds the existing-memory block (``memory_id`` / ``content`` / ...) and
    ``user`` holds the numbered batch of new memories. Each call therefore appends one
    link: new -> supersedes -> current head. Where the head was read is exactly the
    race the per-user lock has to close.
    """
    new_contents = re.findall(r"^\d+\. (.+)$", user, flags=re.MULTILINE)
    if not new_contents:
        return []
    op = {
        "op": "ADD",
        "content": new_contents[0].strip(),
        "layer": "L4_IDENTITY",
        "update_type": "SUPPLEMENT",
    }
    existing_ids = re.findall(r'"memory_id":\s*"([0-9a-f-]{36})"', system)
    if existing_ids:
        op["update_type"] = "OVERRIDE"
        op["supersedes"] = [existing_ids[0]]
    return [op]


def _make_client(settings: Settings, fake_embed, extract) -> MemoryClient:
    return MemoryClient(
        settings=settings,
        embed=fake_embed,
        llm=FakeLLMClient(
            responses={
                "extract": extract,
                "search_query": [],
                "reconcile": _reconcile_supersede_head,
            }
        ),
    )


async def test_same_user_concurrent_add_no_chain_fork(tmp_storage, fake_embed):
    """50 concurrent same-user add() — the evolution chain must stay linear."""
    settings = Settings(mode="system1", storage_dir=tmp_storage)

    counter = {"n": 0}

    def extract(*, system, user):
        counter["n"] += 1
        return _identity_response(counter["n"])

    client = _make_client(settings, fake_embed, extract)

    n_writes = 50
    await asyncio.gather(
        *[
            client.add(content=f"咖啡{i}", app_id="app", user_id="u")
            for i in range(n_writes)
        ]
    )

    where = build_filter(
        app_ids=["app"],
        user_id="u",
        layers=[Layer.L4_IDENTITY],
        statuses=[MemoryStatus.ACTIVE],
    )
    actives = client.factory.vector.get_many(where, limit=1000)

    # Serialized writes thread a single chain: exactly one current head, no fork.
    assert len(actives) == 1, (
        f"expected a linear evolution chain (1 ACTIVE), got {len(actives)} — "
        "concurrent same-user writes forked the chain"
    )
    assert actives[0].is_latest is True

    # Every write is still preserved as a historical version of the chain.
    every_node = client.factory.vector.get_many(
        build_filter(app_ids=["app"], user_id="u", layers=[Layer.L4_IDENTITY]),
        limit=1000,
    )
    assert len(every_node) == n_writes

    await client.aclose()


async def test_cross_user_writes_run_concurrently(tmp_storage, fake_embed):
    """Different users should NOT serialize on each other's write lock."""
    settings = Settings(mode="system1", storage_dir=tmp_storage)

    counter = {"n": 0}

    def extract(*, system, user):
        counter["n"] += 1
        return _identity_response(counter["n"])

    client = _make_client(settings, fake_embed, extract)

    n_users = 5
    await asyncio.gather(
        *[client.add(content=f"事实{i}", app_id="app", user_id=f"u{i}") for i in range(n_users)]
    )

    # Each user gets exactly one chain head — no cross-contamination.
    for i in range(n_users):
        where = build_filter(
            app_ids=["app"],
            user_id=f"u{i}",
            layers=[Layer.L4_IDENTITY],
            statuses=[MemoryStatus.ACTIVE],
        )
        actives = client.factory.vector.get_many(where, limit=10)
        assert len(actives) == 1, f"user u{i} expected 1 active node, got {len(actives)}"

    # Distinct lock keys per user.
    assert len(client._write_locks) == n_users

    await client.aclose()
