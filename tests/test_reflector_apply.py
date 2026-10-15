import pytest
from conftest import FakeJev, FakeLLMClient
from dual_mem import MemoryClient
from dual_mem.config import Settings


@pytest.mark.parametrize(
    "relation", ["OVERRIDE", "NEGATE", "TEMPORAL", "CONFLICT", "SUPPLEMENT"]
)
async def test_public_relation_keeps_history_and_only_replacement_retires_old(
    tmp_storage, fake_embed, relation
):
    jev = FakeJev()
    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient({"extract": {"facts": [{"content": "Old"}]}}),
        jev=jev,
    )
    original = await c.add(content="Old", user_id="u")
    old = original.extra_node_ids[0]
    jev.responses = {"p0:action": "ADD_" + relation, "p0:target": old}
    c.factory.llm.responses = {"extract": {"facts": [{"content": "New"}]}}
    result = await c.add(content="New", user_id="u")
    latest = await c.get(result.extra_node_ids[0])
    ancestor = await c.get(old)
    assert ancestor.content == "Old" and latest.update_type == relation
    assert ancestor.is_latest == (relation not in ("OVERRIDE", "NEGATE"))
    assert latest.conflicts_with == ([old] if relation == "CONFLICT" else [])
    await c.aclose()


async def test_delete_model_archives_without_erasure(tmp_storage, fake_embed):
    jev = FakeJev()
    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient({"extract": {"facts": [{"content": "Old"}]}}),
        jev=jev,
    )
    original = await c.add(content="Old", user_id="u")
    old = original.extra_node_ids[0]
    jev.responses = {"p0:action": "DELETE", "p0:target": old}
    result = await c.add(content="Retract", user_id="u")
    assert result.commit_passed and result.extracted_count == 0
    assert not (await c.get(old)).is_latest and (await c.get(old)).content == "Old"
    await c.aclose()
