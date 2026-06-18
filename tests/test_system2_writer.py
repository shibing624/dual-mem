from conftest import FakeLLMClient

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.system2.system2_writer import System2Writer


def _pending_count(factory) -> int:
    row = factory.cache.conn.execute(
        "SELECT COUNT(*) FROM s2_queue WHERE status = 'pending'"
    ).fetchone()
    return row[0]


def _ultra_factory(tmp_storage, fake_embed):
    return ComponentFactory(
        settings=Settings(mode="ultra", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses={"json": []}),
    )


async def test_ultra_write_enqueues_then_pending_processes(tmp_storage, fake_embed):
    factory = _ultra_factory(tmp_storage, fake_embed)
    writer = System2Writer(factory=factory, agent_mode="full")

    await writer.write(
        content="用户喜欢喝咖啡",
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="",
        request_id="r1",
    )
    assert _pending_count(factory) == 1

    processed = await writer.run_system2_pending()
    assert processed == 1
    assert _pending_count(factory) == 0


async def test_run_system2_pending_empty_queue(tmp_storage, fake_embed):
    factory = _ultra_factory(tmp_storage, fake_embed)
    writer = System2Writer(factory=factory, agent_mode="full")
    assert await writer.run_system2_pending() == 0
