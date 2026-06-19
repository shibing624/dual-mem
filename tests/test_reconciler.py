import pytest

from dual_mem.agent.reconciler import Reconciler
from dual_mem.storage.vector_store import ChromaVectorStore
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


@pytest.fixture
def store(tmp_storage):
    return ChromaVectorStore(tmp_storage)


def _seed(store, fake_embed, content, layer=Layer.L4_IDENTITY):
    node = MemoryNode(
        content=content,
        layer=layer,
        app_id="app",
        user_id="u",
        agent_id="ag",
        status=MemoryStatus.ACTIVE,
        is_latest=True,
    )
    node.embedding = fake_embed.embed_sync(content)
    store.upsert([node])
    return node


async def test_no_candidates_all_add(store, fake_embed):
    llm = FakeLLMClient(responses={"search_query": []})
    rec = Reconciler(llm=llm, embed=fake_embed, vector=store)
    ops = await rec.reconcile(
        new_memories=["用户喜欢喝咖啡"],
        new_memories_meta=[{"content": "用户喜欢喝咖啡", "layer": "L4_IDENTITY", "tags": ["drink"]}],
        app_id="app",
        user_id="u",
        agent_id="ag",
        current_time="",
    )
    assert len(ops) == 1
    assert ops[0].op == "ADD"
    assert ops[0].supersedes == []
    assert ops[0].layer == "L4_IDENTITY"
    assert ops[0].tags == ["drink"]


async def test_supersede_group(store, fake_embed):
    seed = _seed(store, fake_embed, "用户喜欢喝咖啡")
    reconcile_response = [
        {
            "reason": "用户的首选饮品从咖啡变为茶",
            "ops": [
                {
                    "op": "ADD",
                    "content": "用户现在更喜欢喝茶",
                    "layer": "L4_IDENTITY",
                    "supersedes": [seed.node_id],
                    "supersede_reason": "之前喜欢咖啡，现在喜欢茶",
                    "tags": ["DRINK", "Tea"],
                }
            ],
        }
    ]
    llm = FakeLLMClient(responses={"search_query": [], "reconcile": reconcile_response})
    rec = Reconciler(llm=llm, embed=fake_embed, vector=store)

    ops = await rec.reconcile(
        new_memories=["用户喜欢喝咖啡"],
        new_memories_meta=[{"content": "用户喜欢喝咖啡", "layer": "L4_IDENTITY", "tags": ["drink"]}],
        app_id="app",
        user_id="u",
        agent_id="ag",
        current_time="2026-06-18",
    )

    assert len(ops) == 1
    assert ops[0].op == "ADD"
    assert ops[0].content == "用户现在更喜欢喝茶"
    assert ops[0].supersedes == [seed.node_id]
    assert ops[0].supersede_reason == "之前喜欢咖啡，现在喜欢茶"
    assert ops[0].tags == ["drink", "tea"]
    assert ops[0].reason == "用户的首选饮品从咖啡变为茶"


def test_parse_ops_dedup_double_touch():
    data = [
        {
            "reason": "r",
            "ops": [
                {"op": "ADD", "content": "a", "layer": "L2_FACT", "supersedes": ["x"]},
                {"op": "DELETE", "memory_id": "x"},
            ],
        }
    ]
    ops = Reconciler._parse_ops(data)
    assert len(ops) == 1
    assert ops[0].op == "ADD"


def test_parse_ops_empty():
    assert Reconciler._parse_ops([]) == []


def test_parse_ops_unwraps_json_mode_object():
    """JSON mode 下 reconcile 返回 {"updates": [...]} 对象，应被正确解包。"""
    data = {
        "updates": [
            {"reason": "r", "ops": [{"op": "ADD", "content": "用户喜欢茶", "layer": "L4_IDENTITY"}]}
        ]
    }
    ops = Reconciler._parse_ops(data)
    assert len(ops) == 1
    assert ops[0].op == "ADD"
    assert ops[0].content == "用户喜欢茶"


def test_parse_ops_empty_updates_object():
    assert Reconciler._parse_ops({"updates": []}) == []
