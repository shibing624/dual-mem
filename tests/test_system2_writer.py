import pytest

from conftest import FakeLLMClient

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.system2.system2_writer import System2Writer


def _pending_count(factory) -> int:
    row = factory.cache.conn.execute(
        "SELECT COUNT(*) FROM s2_queue WHERE status = 'pending'"
    ).fetchone()
    return row[0]


def _dual_factory(tmp_storage, fake_embed):
    return ComponentFactory(
        settings=Settings(mode="dual", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses={"json": []}),
    )


def _basic_info_only_factory(tmp_storage, fake_embed):
    return ComponentFactory(
        settings=Settings(mode="dual", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(
            responses={
                "extract": {
                    "is_ephemeral": False,
                    "identity": [],
                    "facts": [],
                    "intentions": [],
                    "basic_info": {"name": "Alice"},
                }
            }
        ),
    )


async def test_dual_write_enqueues_then_pending_processes(tmp_storage, fake_embed):
    factory = _dual_factory(tmp_storage, fake_embed)
    writer = System2Writer(factory=factory)

    await writer.write(
        content="用户喜欢喝咖啡",
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="",
        request_id="r1",
    )
    assert _pending_count(factory) == 1

    processed = await writer.digest_pending()
    assert processed == 1
    assert _pending_count(factory) == 0


async def test_digest_pending_empty_queue(tmp_storage, fake_embed):
    factory = _dual_factory(tmp_storage, fake_embed)
    writer = System2Writer(factory=factory)
    assert await writer.digest_pending() == 0


async def test_basic_info_only_write_does_not_enqueue_system2(tmp_storage, fake_embed):
    factory = _basic_info_only_factory(tmp_storage, fake_embed)
    writer = System2Writer(factory=factory)

    await writer.write(
        content="我叫 Alice",
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="s",
        request_id="r1",
    )

    assert factory.cache.list_pending_s2_scopes() == []


async def test_content_hash_hit_does_not_reenqueue_system2(tmp_storage, fake_embed):
    factory = _dual_factory(tmp_storage, fake_embed)
    writer = System2Writer(factory=factory)
    kwargs = {
        "content": "用户喜欢喝咖啡",
        "app_id": "app",
        "user_id": "u",
        "agent_id": "",
        "session_id": "s",
        "request_id": "r1",
    }

    await writer.write(**kwargs)
    await writer.digest_pending()
    await writer.write(**kwargs)

    assert factory.cache.list_pending_s2_scopes() == []


async def test_failed_digest_keeps_scope_pending(tmp_storage, fake_embed, monkeypatch):
    factory = _dual_factory(tmp_storage, fake_embed)
    writer = System2Writer(factory=factory)
    factory.cache.enqueue_s2_task("u", "app")

    async def _fail(**kwargs):
        raise RuntimeError("digest failed")

    monkeypatch.setattr(writer, "_digest_user", _fail)
    with pytest.raises(RuntimeError, match="digest failed"):
        await writer.digest_pending()
    assert factory.cache.list_pending_s2_scopes() == [
        {"app_id": "app", "user_id": "u", "agent_id": ""}
    ]
