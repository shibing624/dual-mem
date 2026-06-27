import pytest

from dual_mem import MemoryClient
from dual_mem.client import _shape_history
from dual_mem.config import Settings
from dual_mem.sdk_models import ChatMessage


def test_shape_history_truncates_assistant_keeps_user_when_over_threshold():
    msgs = [
        ChatMessage(role="user", content="u" * 800),
        ChatMessage(role="assistant", content="a" * 800),
    ]
    # total 1600 chars > threshold 1000 → shaping kicks in
    shaped = _shape_history(msgs, threshold_chars=1000, assistant_max_chars=500)
    assert shaped[0].content == "u" * 800  # user preserved in full
    assert len(shaped[1].content) == 501  # assistant truncated to 500 + ellipsis
    assert shaped[1].content.endswith("…")


def test_shape_history_never_drops_turns():
    # 40 turns batched: all 40 must survive (no turn-count limit anymore).
    msgs = [ChatMessage(role="user", content=f"m{i}") for i in range(40)]
    shaped = _shape_history(msgs, threshold_chars=1, assistant_max_chars=500)
    assert len(shaped) == 40
    assert shaped[0].content == "m0"
    assert shaped[-1].content == "m39"


def test_shape_history_short_dialogue_passthrough():
    # Below threshold → assistant kept in full even if longer than assistant_max_chars cap.
    msgs = [
        ChatMessage(role="user", content="问题"),
        ChatMessage(role="assistant", content="a" * 800),
    ]
    shaped = _shape_history(msgs, threshold_chars=100_000, assistant_max_chars=500)
    assert shaped[1].content == "a" * 800


def test_shape_history_disabled_passthrough():
    msgs = [ChatMessage(role="assistant", content="a" * 800)]
    assert _shape_history(msgs, threshold_chars=0, assistant_max_chars=500)[0].content == "a" * 800
    assert _shape_history(msgs, threshold_chars=1, assistant_max_chars=0)[0].content == "a" * 800


@pytest.fixture
def client(tmp_storage, fake_embed, fake_llm):
    return MemoryClient(
        storage_dir=tmp_storage, mode="system1", embed=fake_embed, llm=fake_llm
    )


async def test_add_drops_system_role_from_extract_dialogue(client, fake_llm):
    await client.add(
        messages=[
            {"role": "system", "content": "You are a helpful assistant with secret rules."},
            {"role": "user", "content": "用户喜欢咖啡"},
            {"role": "assistant", "content": "好的，记住了。"},
        ],
        app_id="app",
        user_id="u",
    )
    extract_calls = [
        c
        for c in fake_llm.calls
        if c["type"] == "chat_json"
        and ("memory analyst" in c["system"] or "记忆分析专家" in c["system"])
    ]
    assert extract_calls
    payload = extract_calls[0]["user"]
    assert "secret rules" not in payload
    assert "[system]" not in payload
    assert "用户喜欢咖啡" in payload
    assert "好的" in payload


async def test_add_then_search_hit(client):
    added = await client.add(content="用户喜欢喝咖啡", app_id="app", user_id="u")
    assert added.success is True
    memory_id = added.memory_id

    found = await client.search(
        query="用户喜欢喝咖啡", app_ids=["app"], user_id="u", min_score=0.4
    )
    assert found.success is True
    normal = found.memories.normal
    assert any("咖啡" in item.content for item in normal)


async def test_get_and_list(client):
    added = await client.add(content="用户住在北京", app_id="app", user_id="u")
    memory_id = added.memory_id

    got = await client.get(memory_id)
    assert got is not None
    assert got.content == "用户住在北京"

    listed = await client.list(app_id="app", user_id="u")
    assert len(listed) >= 1


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
    assert done.deleted >= 2


async def test_add_search_use_default_app_id(tmp_storage, fake_embed, fake_llm):
    client = MemoryClient(
        storage_dir=tmp_storage,
        mode="system1",
        embed=fake_embed,
        llm=fake_llm,
        settings=Settings(
            default_app_id="my_tenant",
            llm_api_key="x",
            embed_api_key="y",
        ),
    )
    added = await client.add(content="默认租户记忆", user_id="u")
    assert added.success is True

    found = await client.search(query="默认租户记忆", user_id="u", min_score=0.4)
    assert found.success is True
    assert any("默认租户" in item.content for item in found.memories.normal)


async def test_from_config(tmp_storage, fake_embed, fake_llm, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("DUAL_MEM_CONFIG_FILE", str(Path(tmp_storage) / "nope.yaml"))
    client = MemoryClient.from_config(
        {
            "storage_dir": tmp_storage,
            "default_app_id": "cfg_app",
            "llm": {"api_key": "k", "model": "m"},
            "embedder": {"api_key": "e", "model": "em"},
        },
        embed=fake_embed,
        llm=fake_llm,
    )
    assert client.settings.default_app_id == "cfg_app"
    assert client.settings.storage_dir == tmp_storage

    added = await client.add(content="from_config 写入", user_id="u")
    listed = await client.list(user_id="u")
    assert len(listed) >= 1
