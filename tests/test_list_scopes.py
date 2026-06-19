import pytest

from dual_mem import MemoryClient
from dual_mem.config import Settings


async def test_list_scopes(tmp_storage, fake_embed, fake_llm):
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage, gate_enabled=False),
        embed=fake_embed,
        llm=fake_llm,
    )
    await client.add(content="alpha", app_id="app_a", user_id="u1")
    await client.add(content="beta", app_id="app_a", user_id="u2")

    scopes = await client.list_scopes(app_id="app_a")
    users = {s.user_id for s in scopes}
    assert users == {"u1", "u2"}
    assert all(s.app_id == "app_a" for s in scopes)
