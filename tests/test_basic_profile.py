import pytest

from dual_mem.agent.basic_profile import (
    BASIC_FIELDS,
    BasicProfileTool,
    normalize_basic_info,
    render_content,
)
from dual_mem.isolation import build_filter
from dual_mem.storage.vector_store import ChromaVectorStore
from dual_mem.types import Layer, MemoryStatus


@pytest.fixture
def store(tmp_storage):
    return ChromaVectorStore(tmp_storage)


def _full_kv(nodes):
    nodes = sorted(nodes, key=lambda n: n.gmt_created)
    kv = {}
    for node in nodes:
        for k, v in (node.custom or {}).get("basic_info_kv", {}).items():
            if k in BASIC_FIELDS:
                kv[k] = v
    return kv


def test_render_content():
    assert render_content({"name": "张三", "age": 30}) == "The user's name is 张三, age is 30."
    assert render_content({}) == ""


def test_normalize_basic_info_rejects_unknown_empty_and_malformed_fields():
    assert normalize_basic_info(
        {
            "name": {},
            "age": [],
            "location": "  北京  ",
            "occupation": None,
            "employer": "null",
            "nickname": "小明",
        }
    ) == {"location": "北京"}
    assert normalize_basic_info(None) == {}


async def test_evolution_chain_two_applies(store, fake_embed):
    tool = BasicProfileTool(vector=store, embed=fake_embed)

    id1 = await tool.apply(
        arguments={"name": "张三"},
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
    )
    assert id1 is not None

    id2 = await tool.apply(
        arguments={"location": "北京"},
        app_id="app",
        user_id="u",
        agent_id="ag",
        session_id="se",
    )
    assert id2 is not None and id2 != id1

    head = store.get(id2)
    assert head.is_latest is True
    assert head.status is MemoryStatus.ACTIVE
    assert head.supersedes == [id1]
    assert head.custom["basic_info_kv"] == {"location": "北京"}

    old = store.get(id1)
    assert old.is_latest is False
    assert old.status is MemoryStatus.SUPERSEDED
    assert old.superseded_by == [id2]

    where = build_filter(app_ids=["app"], user_id="u", agent_ids=["ag"], layers=[Layer.L0_BASIC_INFO])
    all_l0 = store.get_many(where)
    assert _full_kv(all_l0) == {"name": "张三", "location": "北京"}


async def test_no_diff_returns_none(store, fake_embed):
    tool = BasicProfileTool(vector=store, embed=fake_embed)
    await tool.apply(arguments={"name": "张三"}, app_id="app", user_id="u", agent_id="ag", session_id="se")
    again = await tool.apply(arguments={"name": "张三"}, app_id="app", user_id="u", agent_id="ag", session_id="se")
    assert again is None


async def test_empty_arguments_returns_none(store, fake_embed):
    tool = BasicProfileTool(vector=store, embed=fake_embed)
    assert await tool.apply(arguments={"name": " "}, app_id="app", user_id="u", agent_id="ag", session_id="se") is None
