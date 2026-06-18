import pytest

from dual_mem.agent.reconciler import Reconciler, ReconcileOp
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
    node.embedding = fake_embed.embed(content)
    store.upsert([node])
    return node


def test_no_candidates_all_add(store, fake_embed):
    llm = FakeLLMClient(responses={"search_query": []})
    rec = Reconciler(llm=llm, embed=fake_embed, vector=store)
    ops = rec.reconcile(
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


def test_supersede_group(store, fake_embed):
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

    ops = rec.reconcile(
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
