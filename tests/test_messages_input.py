"""Structured messages input is normalized and sent intact to L1 and Extractor."""
from dual_mem.client import MemoryClient, _format_dialogue, _normalize_messages
from dual_mem.config import Settings
from dual_mem.sdk_models import ChatMessage

from conftest import FakeLLMClient


_EMPTY_EXTRACT = {
    "is_ephemeral": False,
    "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
    "identity": [],
    "facts": [],
    "intentions": [],
    "basic_info": {},
}


def test_normalize_messages_dict_and_chatmessage_mixed():
    """dict 和 ChatMessage 混用，role 默认 user，content 空白丢弃。"""
    out = _normalize_messages([
        {"role": "user", "content": "hi"},
        ChatMessage(role="assistant", content="hello"),
        {"role": "user", "content": ""},        # dropped
        {"role": "system", "content": "  "},    # dropped
        {"content": "无 role"},                  # role 默认 user
    ])
    assert [(m.role, m.content) for m in out] == [
        ("user", "hi"),
        ("assistant", "hello"),
        ("user", "无 role"),
    ]


def test_format_dialogue_uses_role_labels():
    msgs = _normalize_messages([
        {"role": "user", "content": "今天累死了"},
        {"role": "assistant", "content": "辛苦了"},
        {"role": "user", "content": "明天还要加班"},
    ])
    text = _format_dialogue(msgs)
    assert text == "[user]: 今天累死了\n[assistant]: 辛苦了\n[user]: 明天还要加班"


async def test_messages_input_reaches_l1_and_extractor(tmp_storage, fake_embed):
    """messages 输入：L1 和 Extractor 都接收保留角色标记的完整对话。"""
    settings = Settings(mode="system1", storage_dir=tmp_storage)

    captured: dict = {}

    def extract_capture(*, system, user):
        captured["dialogue"] = user
        return _EMPTY_EXTRACT

    client = MemoryClient(
        settings=settings,
        embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": extract_capture, "search_query": []}),
    )

    result = await client.add(
        messages=[
            {"role": "user", "content": "我刚搬到北京，准备入职新公司"},
            {"role": "assistant", "content": "恭喜！什么公司？"},
            {"role": "user", "content": "一家做记忆系统的创业团队"},
        ],
        app_id="app",
        user_id="u_messages",
    )

    assert result.success
    # L1_RAW 存的是带 role 标记的完整对话
    raw = client.factory.vector.get(result.memory_id)
    assert raw is not None
    assert "[user]: 我刚搬到北京" in raw.content
    assert "[assistant]: 恭喜！" in raw.content
    # Extractor 看到的也是完整对话
    assert "[user]:" in captured["dialogue"]
    assert "[assistant]:" in captured["dialogue"]

    await client.aclose()


async def test_messages_input_preserves_all_messages_as_current(tmp_storage, fake_embed):
    settings = Settings(
        mode="system1",
        storage_dir=tmp_storage,
    )
    captured: dict = {}

    def extract_capture(*, system, user):
        captured["prompt"] = user
        return _EMPTY_EXTRACT

    client = MemoryClient(
        settings=settings,
        embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": extract_capture, "search_query": []}),
    )

    await client.add(
        messages=[
            {"role": "user", "content": "old context"},
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "new fact"},
        ],
        app_id="app",
        user_id="u_window",
    )

    prompt = captured["prompt"]
    history, current = prompt.split("Current messages", maxsplit=1)
    assert "old context" not in history
    assert "old context" in current
    assert "recent answer" in current
    assert "new fact" in current
    assert "never extract new facts" in prompt

    await client.aclose()


async def test_messages_input_compatible_with_chat_message_objects(tmp_storage, fake_embed):
    """ChatMessage dataclass 直接传也能跑通。"""
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(
        settings=settings,
        embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": _EMPTY_EXTRACT, "search_query": []}),
    )

    result = await client.add(
        messages=[
            ChatMessage(role="user", content="我喜欢喝咖啡"),
            ChatMessage(role="assistant", content="OK"),
        ],
        app_id="app",
        user_id="u_chatmsg",
    )
    assert result.success
    raw = client.factory.vector.get(result.memory_id)
    assert raw is not None
    assert raw.content == "[user]: 我喜欢喝咖啡\n[assistant]: OK"

    await client.aclose()
