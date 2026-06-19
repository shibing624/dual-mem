import pytest

from dual_mem import MemoryClient


@pytest.fixture
def client(tmp_storage, fake_embed, fake_llm):
    return MemoryClient(
        storage_dir=tmp_storage, mode="system1", embed=fake_embed, llm=fake_llm
    )


async def test_add_then_search_hit(client):
    added = await client.add(content="用户喜欢喝咖啡", app_id="app", user_id="u")
    assert added.success is True
    memory_id = added.memory_id

    found = await client.search(
        query="用户喜欢喝咖啡", app_ids=["app"], user_id="u", min_score=0.4
    )
    assert found.success is True
    normal = found.memories.normal
    assert any(item.memory_id == memory_id for item in normal)


async def test_get_and_list(client):
    added = await client.add(content="用户住在北京", app_id="app", user_id="u")
    memory_id = added.memory_id

    got = await client.get(memory_id)
    assert got is not None
    assert got.content == "用户住在北京"

    listed = await client.list(app_id="app", user_id="u")
    assert any(item.memory_id == memory_id for item in listed)


async def test_update(client):
    added = await client.add(content="旧内容", app_id="app", user_id="u")
    memory_id = added.memory_id

    res = await client.update(memory_id, "新内容")
    assert res.success is True
    got = await client.get(memory_id)
    assert got.content == "新内容"


async def test_delete_and_missing_404(client):
    added = await client.add(content="待删除", app_id="app", user_id="u")
    memory_id = added.memory_id

    res = await client.delete(memory_id)
    assert res.success is True
    assert await client.get(memory_id) is None

    missing = await client.delete("no-such-id")
    assert missing.success is False
    assert missing.error_code == 404


async def test_delete_bulk_confirm_guard(client):
    await client.add(content="批量1", app_id="app", user_id="u")
    await client.add(content="批量2", app_id="app", user_id="u")

    guarded = await client.delete_bulk(app_id="app", user_id="u")
    assert guarded.success is False
    assert guarded.error_code == 400

    done = await client.delete_bulk(app_id="app", user_id="u", confirm=True)
    assert done.success is True
    assert done.deleted == 2
