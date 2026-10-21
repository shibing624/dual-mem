import json

from conftest import FakeJev, FakeLLMClient

from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode


def _vec(member: int) -> list[float]:
    v = [0.0] * 64
    v[0], v[1 + member] = 1.0, 0.5
    return v


async def test_dual_schema_digest_recall_and_exclusion(tmp_storage, fake_embed):
    def response(system, user):
        if "cognitive processing agent" not in system:
            return {"domains": {}} if "persona catalog" in system else {"abstraction": "x"}
        mids = [m["id"] for m in json.loads(user)["cluster"]]
        return {"operations": [{
            "op": "create_schema",
            "content": "When preparing talks, the user rehearses with diagrams — reflecting a need for clarity.",
            "evidence_ids": mids,
        }]}

    c = MemoryClient(
        settings=Settings(mode="dual", storage_dir=tmp_storage, derived_context_limit=10),
        embed=fake_embed,
        llm=FakeLLMClient({"json": response}),
        jev=FakeJev(),
    )
    facts = [
        MemoryNode(node_id=f"f{i}", content=f"User rehearsed talk {i} with diagrams",
                   layer=Layer.L1_FACT, app_id="default", user_id="u", embedding=_vec(i),
                   custom={"source_node_id": f"raw{i}"})
        for i in range(4)
    ]
    c.factory.vector.upsert(facts)
    c.factory.cache.enqueue_s2_task("u", "default")
    assert (await c.digest()).processed == 1
    (schema,) = c.factory.vector.get_many({"layer": Layer.L3_DERIVED.value})
    assert schema.custom["raw_source_ids"] == ["raw0", "raw1", "raw2", "raw3"]

    found = await c.search(query="User rehearsed talk 0 with diagrams", user_id="u",
                           app_ids=["default"], min_score=0)
    assert [m.memory_id for m in found.memories.schema] == [schema.node_id]
    assert set(found.memories.schema[0].supporting_ids) <= {f"f{i}" for i in range(4)}
    excluded = await c.search(query="User rehearsed talk 0 with diagrams", user_id="u",
                              app_ids=["default"], min_score=0, include_derived=False)
    assert excluded.memories.schema == []
    assert all(m.layer != Layer.L3_DERIVED.value for m in excluded.memories.flatten())
    await c.aclose()
