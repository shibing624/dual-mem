"""Demo 6 — Coding 记忆：含 Tool 调用的工程排障对话沉淀、版本演化与检索。

开启 `coding_enabled=True` 后，当 `add(messages=[...])` 包含 tool 调用或 `role="tool"` 消息时：
1. 先将完整工具会话落为不可丢的 L0 原始证据；
2. JEV 判定对话是否属于工程研发场景（CODING vs CHAT）；
3. CodingExtractor 抽取结构化经验（`task`、`solution`、`search_keys`、`boundary_envs`、`files`）；
4. JEV 在相同作用域与版本边界内执行 target-first 关系裁决（新增 / 覆盖 / 合并 / 跳过 / 归档）；
5. 通过 `search_coding()` 只检索当前生效的工程经验，被覆盖的旧方案保留在历史里。

环境要求：jev_*、llm_*、embed_* 都要有。
运行：python examples/06_coding_memory.py
"""

import asyncio
import json

from _common import example_settings, section, show_write

from dual_mem import MemoryClient

USER = "dev_alice"


def tool_session(
    *, ask: str, call_id: str, tool: str, args: dict, output: str, answer: str
) -> list[dict]:
    """一轮标准 OpenAI 格式的工具会话：user → tool_calls → tool 结果 → 结论。"""
    return [
        {"role": "user", "content": ask},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(args)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": output},
        {"role": "assistant", "content": answer},
    ]


async def show_hits(client: MemoryClient, query: str) -> None:
    print(f"\n  Query: {query}")
    for item in await client.search_coding(query=query, user_id=USER, top_k=3):
        print(
            f"    - [score={item['score']:.2f}] task: {item['task']}\n"
            f"      solution: {item['solution']}\n"
            f"      files: {item['files']}  supersedes: {item['supersedes']}"
        )


async def main() -> None:
    settings = example_settings("system1", "coding", coding_enabled=True)
    client = MemoryClient(settings=settings)
    try:
        section("第 1 轮：SQLite 并发写入报 database is locked")
        res = await client.add(
            messages=tool_session(
                ask="app/db.py 在高并发写入时偶发 sqlite3.OperationalError: database is locked，帮我定位并修复。",
                call_id="call_1",
                tool="read_file",
                args={"path": "app/db.py"},
                output=(
                    "conn = sqlite3.connect(db_path, timeout=5.0)\n"
                    "conn.execute('SELECT ...')  # deferred transaction, upgrades to write later"
                ),
                answer=(
                    "原因：DEFERRED 事务先读后写，升级写锁时若另一连接已持有写锁，"
                    "SQLite 会直接返回 SQLITE_BUSY 而不等待 timeout。"
                    "修复：写事务显式 BEGIN IMMEDIATE 提前拿写锁，并把 timeout 调到 30s。"
                ),
            ),
            user_id=USER,
            session_id="s1",
        )
        show_write(res, "写入 SQLite 锁冲突排障经验")

        section("第 2 轮：pytest 全量测试卡死（独立的另一条经验）")
        res = await client.add(
            messages=tool_session(
                ask="跑全量 pytest 时在 50% 左右卡住，看看 tests/conftest.py 怎么处理 chromadb client 的。",
                call_id="call_2",
                tool="read_file",
                args={"path": "tests/conftest.py"},
                output="SharedSystemClient._identifier_to_system accumulates hundreds of Rust bindings.",
                answer=(
                    "在 tests/conftest.py 的 autouse fixture 中，每个测试结束后调用 "
                    "SharedSystemClient.clear_system_cache() 释放 Rust 句柄，全量测试不再卡死。"
                ),
            ),
            user_id=USER,
            session_id="s2",
        )
        show_write(res, "写入 pytest 释放 Chroma 缓存经验")

        section("第 3 轮：方案演化——BEGIN IMMEDIATE 仍会排队超时，换成 WAL")
        res = await client.add(
            messages=tool_session(
                ask="app/db.py 改成 BEGIN IMMEDIATE 后，读多写少的高峰期写请求还是排队到 30s 超时。",
                call_id="call_3",
                tool="run_shell",
                args={"command": "sqlite3 data/app.db 'PRAGMA journal_mode;'"},
                output="delete",
                answer=(
                    "确认数据库仍是 rollback journal（journal_mode=delete），读写互斥。"
                    "替换之前的方案：app/db.py 打开连接后执行 PRAGMA journal_mode=WAL 和 "
                    "PRAGMA busy_timeout=30000，读不再阻塞写；BEGIN IMMEDIATE 只保留在写事务上。"
                ),
            ),
            user_id=USER,
            session_id="s3",
        )
        show_write(res, "写入新方案（期望 ADD/OVERRIDE 覆盖第 1 轮）")

        section("search_coding：只返回当前生效的方案")
        await show_hits(client, "SQLite 高并发写入报 database is locked 怎么解决？")
        await show_hits(client, "pytest 全量测试卡死在 chromadb 怎么办？")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
