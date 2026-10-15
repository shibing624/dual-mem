import json
from conftest import FakeLLMClient, FakeJev
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer


async def test_dual_evidence_derived_recall_and_history(tmp_storage, fake_embed):
    def response(system, user):
        ids = [f["id"] for f in json.loads(user)["facts"]]
        return {
            "memories": [
                {
                    "content": "User prepares carefully",
                    "kind": "schema",
                    "evidence_ids": ids,
                },
                {
                    "content": "User will give a talk",
                    "kind": "intention",
                    "evidence_ids": ids,
                },
            ],
            "processed_ids": ids,
        }

    c = MemoryClient(
        settings=Settings(mode="dual", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient({"json": response}),
        jev=FakeJev(),
    )
    written = await c.add(content="User prepares talk with diagrams", user_id="u")
    assert c.factory.cache.reconcile_queue_size() == 0
    assert (await c.digest()).processed == 1
    found = await c.search(
        query="User prepares carefully",
        user_id="u",
        min_score=0,
        profile_min_score=0,
        intention_limit=5,
    )
    assert any(
        m.content == "User prepares carefully" and m.kind == "schema"
        for m in found.memories.profile
    )
    assert any(m.kind == "intention" for m in found.memories.proactive)
    excluded = await c.search(
        query="User prepares carefully",
        user_id="u",
        min_score=0,
        profile_min_score=0,
        include_derived=False,
    )
    assert all(m.layer != Layer.L3_DERIVED.value for m in excluded.memories.profile)
    old = written.extra_node_ids[0]
    updated = await c.update(old, "User cancelled talk")
    assert updated.memory_id != old and not (await c.get(old)).is_latest
    derived = c.factory.vector.get_many({"layer": Layer.L3_DERIVED.value})
    assert all(not n.is_latest for n in derived)
    assert all(n.content for n in derived)
    await c.aclose()
