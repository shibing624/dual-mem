import pytest

from dual_mem.config import Settings
from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryStatus
from dual_mem.writer.memory_writer import MemoryWriter


@pytest.fixture
def factory(tmp_storage, fake_embed):
    f = ComponentFactory(settings=Settings(mode="lite", storage_dir=tmp_storage))
    f._embed = fake_embed
    return f


async def test_lite_write_single_raw_active(factory):
    writer = MemoryWriter(factory=factory, agent_mode="disabled")
    result = await writer.write(
        content="用户喜欢喝咖啡",
        app_id="app",
        user_id="u",
        agent_id="",
        session_id="",
        request_id="req-1",
    )

    where = build_filter(app_ids=["app"], user_id="u")
    nodes = factory.vector.get_many(where)
    assert len(nodes) == 1
    assert nodes[0].node_id == result.memory_id
    assert nodes[0].layer is Layer.L1_RAW
    assert nodes[0].status is MemoryStatus.ACTIVE
    assert result.extra_node_ids == []
