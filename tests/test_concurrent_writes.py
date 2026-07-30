"""Concurrency regression for the per-user write lock.

Same-user concurrent add() must serialize through the fast-write -> reconcile pipeline,
otherwise the evolution chain forks and `is_latest` count diverges from ADD count.
Different users must NOT block each other (write lock is keyed per user).
"""
import asyncio

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


async def test_same_user_concurrent_add_no_chain_fork(tmp_storage, fake_embed):
    """50 concurrent same-user add() — is_latest == ACTIVE node count (no fork)."""
    settings = Settings(mode="system1", storage_dir=tmp_storage)

    counter = {"n": 0}

    def extract(*, system, user):
        counter["n"] += 1
        return _identity_response(counter["n"])

    client = MemoryClient(
        settings=settings,
        embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": extract, "search_query": []}),
    )

    n_writes = 50
    await asyncio.gather(
        *[
            client.add(content=f"咖啡{i}", app_id="app", user_id="u")
            for i in range(n_writes)
        ]
    )

    # Each add() yields exactly one identity node from the scripted extractor response.
    where = build_filter(
        app_ids=["app"],
        user_id="u",
        layers=[Layer.L4_IDENTITY],
        statuses=[MemoryStatus.ACTIVE],
    )
    actives = client.factory.vector.get_many(where, limit=1000)
    assert len(actives) == n_writes
    # Every active node must be the chain head.
    assert all(node.is_latest for node in actives)

    await client.aclose()


async def test_cross_user_writes_run_concurrently(tmp_storage, fake_embed):
    """Different users should NOT serialize on each other's write lock."""
    settings = Settings(mode="system1", storage_dir=tmp_storage)

    counter = {"n": 0}

    def extract(*, system, user):
        counter["n"] += 1
        return _identity_response(counter["n"])

    client = MemoryClient(
        settings=settings,
        embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": extract, "search_query": []}),
    )

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
