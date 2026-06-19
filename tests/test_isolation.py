from dual_mem.isolation import build_filter, isolation_key
from dual_mem.types import Layer, MemoryStatus


def test_isolation_key_full():
    assert isolation_key("u", "ag", "se") == "u::ag::se"


def test_isolation_key_defaults():
    assert isolation_key("u") == "u::::"


def test_build_filter_minimal():
    f = build_filter(app_ids=["a1", "a2"], user_id="u")
    assert f == {"app_id": {"$in": ["a1", "a2"]}, "user_id": "u"}


def test_build_filter_all_options():
    f = build_filter(
        app_ids=["a1"],
        user_id="u",
        agent_ids=["ag1"],
        session_ids=["s1", "s2"],
        layers=[Layer.L2_FACT],
        statuses=[MemoryStatus.ACTIVE],
    )
    assert f["app_id"] == {"$in": ["a1"]}
    assert f["user_id"] == "u"
    assert f["agent_id"] == {"$in": ["ag1"]}
    assert f["session_id"] == {"$in": ["s1", "s2"]}
    assert f["layer"] == {"$in": ["L2_FACT"]}
    assert f["status"] == {"$in": ["ACTIVE"]}


def test_build_filter_omits_none():
    f = build_filter(app_ids=["a1"], user_id="u", agent_ids=None, layers=None)
    assert "agent_id" not in f
    assert "layer" not in f
    assert "status" not in f


async def test_fixtures_available(fake_embed, fake_llm, tmp_storage):
    v1 = fake_embed.embed_sync("hello")
    v2 = fake_embed.embed_sync("hello")
    assert v1 == v2
    assert len(v1) == 64
    batch = await fake_embed.embed_batch(["a", "b"])
    assert len(batch) == 2
    out = await fake_llm.chat_json(system="s", user="u")
    assert out == {"facts": [], "identity": []}
    assert isinstance(tmp_storage, str)
