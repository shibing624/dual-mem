import asyncio
from conftest import FakeJev, FakeLLMClient
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer


async def test_parallel_same_scope_cas_retry_leaves_one_linear_latest_chain(
    tmp_storage, fake_embed
):
    def script(state, questions):
        heads = state.get("current", [])
        return {
            "p0:action": "ADD_OVERRIDE" if heads else "ADD_SUPPLEMENT",
            "p0:target": heads[0]["id"] if heads else "NONE",
        }

    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(),
        jev=FakeJev(script),
    )
    await asyncio.gather(
        *(c.add(content=f"current state {i}", user_id="u") for i in range(8))
    )
    facts = c.factory.vector.get_many(
        {"user_id": "u", "layer": Layer.L1_FACT.value}, limit=100
    )
    assert len(facts) == 8
    assert sum(n.is_latest for n in facts) == 1
    assert sum(bool(n.supersedes) for n in facts) == 7
    assert all(n.status.value == "ACTIVE" for n in facts)


async def test_independent_scopes_overlap_decision_io(tmp_storage, fake_embed):
    class SlowJev(FakeJev):
        active = peak = 0

        async def evaluate(self, **kw):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.02)
            try:
                return await super().evaluate(**kw)
            finally:
                self.active -= 1

    jev = SlowJev()
    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(),
        jev=jev,
    )
    await asyncio.gather(*(c.add(content="state", user_id=str(i)) for i in range(4)))
    assert jev.peak == 4
