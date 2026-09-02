"""Host-injected memory blocks must not be written back as L1 / extract input."""
from dual_mem.client import MemoryClient, strip_memory_injection
from dual_mem.config import Settings

from conftest import FakeLLMClient

_INJECTED = (
    "<relevant-memories>\n"
    "The following are stored memories for the current user.\n"
    "- [1] 用户喜欢咖啡\n"
    "</relevant-memories>\n"
    "我今天想喝茶"
)


def test_strip_memory_injection_drops_known_blocks():
    text = (
        "<user-profile>\n- 姓名: 张三\n</user-profile>\n"
        f"{_INJECTED}\n"
        "<memory-tools-guide>\n当上方记忆不足时调用 memory_search。\n"
        "</memory-tools-guide>\n"
        "<topic-catalog>\n- 工作\n</topic-catalog>"
    )
    cleaned = strip_memory_injection(text)
    assert "用户喜欢咖啡" not in cleaned
    assert "张三" not in cleaned
    assert "memory_search" not in cleaned
    assert "工作" not in cleaned
    assert "我今天想喝茶" in cleaned


def test_strip_memory_injection_noop_on_plain_text():
    assert strip_memory_injection("用户住在上海") == "用户住在上海"


async def test_add_strips_injection_from_messages(tmp_storage, fake_embed):
    captured: dict = {}

    def extract_capture(*, system, user):
        captured["dialogue"] = user
        return {
            "is_ephemeral": False,
            "identity": [],
            "facts": [],
            "intentions": [],
            "basic_info": {},
        }

    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": extract_capture, "search_query": []}),
    )
    result = await client.add(
        messages=[{"role": "user", "content": _INJECTED}],
        app_id="app",
        user_id="u_inject",
    )
    assert result.success
    raw = client.factory.vector.get(result.memory_id)
    assert raw is not None
    assert "relevant-memories" not in raw.content
    assert "用户喜欢咖啡" not in raw.content
    assert "我今天想喝茶" in raw.content
    assert "用户喜欢咖啡" not in captured["dialogue"]
    assert "我今天想喝茶" in captured["dialogue"]
    await client.aclose()


async def test_add_injection_only_does_not_write(tmp_storage, fake_embed):
    client = MemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=FakeLLMClient(),
    )
    result = await client.add(
        content="<relevant-memories>\n- [1] 脏记忆\n</relevant-memories>",
        app_id="app",
        user_id="u_empty",
    )
    assert result.success
    assert result.memory_id == ""
    assert result.commit_passed is False
    nodes = client.factory.vector.get_many(
        {"$and": [{"app_id": "app"}, {"user_id": "u_empty"}]},
        limit=10,
    )
    assert nodes == []
    await client.aclose()
