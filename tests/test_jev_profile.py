from conftest import FakeJev, FakeLLMClient
from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer


async def test_new_profile_dimension_does_not_retire_other_dimension(
    tmp_storage, fake_embed
):
    jev = FakeJev({"p0:representation": "PROFILE"})
    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient({"extract": {"basic_info": {"name": "Alice"}}}),
        jev=jev,
    )
    named = await c.add(content="My name is Alice", user_id="u")
    c.factory.llm.responses = {"extract": {"basic_info": {"location": "Beijing"}}}
    located = await c.add(content="Lives in Beijing", user_id="u")
    assert (await c.get(named.extra_node_ids[0])).is_latest
    assert (await c.get(located.extra_node_ids[0])).is_latest
    assert (await c.get(located.extra_node_ids[0])).layer == Layer.L2_PROFILE.value


async def test_profile_update_links_only_corresponding_dimension(
    tmp_storage, fake_embed
):
    jev = FakeJev({"p0:representation": "PROFILE"})
    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient({"extract": {"basic_info": {"location": "Beijing"}}}),
        jev=jev,
    )
    located = await c.add(content="Lives in Beijing", user_id="u")
    old = located.extra_node_ids[0]
    c.factory.llm.responses = {"extract": {"basic_info": {"location": "Shanghai"}}}
    jev.responses = {
        "p0:representation": "PROFILE",
        "p0:action": "ADD_OVERRIDE",
        "p0:target": old,
    }
    updated = await c.add(content="Moved to Shanghai", user_id="u")
    assert updated.decisions[0].update_type == "OVERRIDE" and updated.decisions[
        0
    ].targets == [old]
    assert not (await c.get(old)).is_latest
    assert (await c.get(updated.extra_node_ids[0])).supersedes == [old]
