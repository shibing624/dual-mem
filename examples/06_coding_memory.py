"""Demo 6 — Coding 记忆：含 Tool 调用的工程排障对话沉淀与检索。

开启 `coding_enabled=True` 后，当 `add(messages=[...])` 包含 `role="tool"` 消息时：
1. 先将完整工具会话落为不可丢的 L0 原始证据；
2. JEV 判定对话是否属于工程研发场景（CODING vs CHAT）；
3. CodingExtractor 抽取结构化经验（`task`、`solution`、`search_keys`、`boundary_envs`、`files`）；
4. JEV 在相同作用域与边界内执行 target-first 关系裁决（新增 / 覆盖 / 合并 / 跳过 / 归档）；
5. 通过 `search_coding()` 检索最新生效的工程经验。

环境要求：jev_*、llm_*、embed_* 都要有。
运行：python examples/06_coding_memory.py
"""

import asyncio

from _common import example_settings, section, show_write

from dual_mem import MemoryClient


async def main() -> None:
    settings = example_settings("system1", "coding", coding_enabled=True)
    client = MemoryClient(settings=settings)
    user = "dev_alice"
    try:
        section("第 1 轮：含 Tool 输出的排障对话（SQLite 并发锁冲突）")
        session_1 = [
            {
                "role": "user",
                "content": "dual_mem/storage/vector_store.py 在高并发写入时偶发 sqlite3.OperationalError: database is locked，帮我定位并修复。",
            },
            {
                "role": "assistant",
                "content": "我先检查一下 dual_mem/storage/vector_store.py 里的事务开启方式。",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "dual_mem/storage/vector_store.py"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": (
                    "File: dual_mem/storage/vector_store.py\n"
                    "conn = sqlite3.connect(db_path, timeout=5.0)\n"
                    "# default deferred transaction upgrades lock mid-flight"
                ),
            },
            {
                "role": "assistant",
                "content": "已确认原因：默认 DEFERRED 事务在读后升级写锁会撞上死锁。修复方案是显式执行 BEGIN IMMEDIATE 提前拿写锁，并将连接 timeout 调到 30s。",
            },
        ]
        res1 = await client.add(messages=session_1, user_id=user, session_id="s1")
        show_write(res1, "写入 SQLite 并发锁排障记录")

        section("第 2 轮：含 Tool 输出的构建环境经验（Chroma 缓存释放）")
        session_2 = [
            {
                "role": "user",
                "content": "跑全量 pytest 时在 50% 左右卡住，看看 tests/conftest.py 怎么处理 chromadb client 的。",
            },
            {
                "role": "assistant",
                "content": "我先读取 tests/conftest.py。",
                "tool_calls": [
                    {
                        "name": "read_file",
                        "arguments": {"path": "tests/conftest.py"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": (
                    "File: tests/conftest.py\n"
                    "SharedSystemClient._identifier_to_system accumulates hundreds of Rust bindings."
                ),
            },
            {
                "role": "assistant",
                "content": "在 tests/conftest.py 的 autouse fixture 中，每个测试结束后调用 SharedSystemClient.clear_system_cache() 即可释放 Rust 句柄，避免全量测试卡死。",
            },
        ]
        res2 = await client.add(messages=session_2, user_id=user, session_id="s2")
        show_write(res2, "写入 pytest 释放 Chroma 缓存经验")

        section("search_coding：检索工程排障经验")
        for query in [
            "SQLite 高并发写入报 database is locked 怎么解决？",
            "pytest 全量测试卡死在 chromadb 怎么办？",
        ]:
            print(f"\n  Query: {query}")
            hits = await client.search_coding(query=query, user_id=user, top_k=3)
            for item in hits:
                print(
                    f"    - [score={item['score']:.2f}] task: {item['task']}\n"
                    f"      solution: {item['solution']}\n"
                    f"      files: {item['files']}  scope: {item['boundary_scope']}"
                )
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
