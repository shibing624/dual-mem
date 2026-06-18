"""Demo 2 — pro 档：System1 抽取 + 演化链。

场景：用户偏好随时间变化，记忆系统应自动用新认知"取代"旧认知，并保留可追溯的演化链。
pro 档每次写入会跑 LLM：抽取 identity/facts、L0 基础画像工具、Reconciler 整理
（ADD / SUPERSEDE / DELETE）、长文本摘要。

运行：python examples/02_pro.py
"""
import asyncio

from _common import fresh_storage, section, show_memories

from dual_mem import MemoryClient


async def main() -> None:
    client = MemoryClient(mode="pro", storage_dir=fresh_storage("pro"))
    app, user = "default", "bob"

    section("第 1 轮写入：自我介绍 + 旧偏好")
    for text in [
        "我叫王磊，今年 30 岁，是一名后端工程师，在上海工作。",
        "我最喜欢的编程语言是 Java，已经用了五年了，非常熟练。",
    ]:
        res = await client.add(content=text, app_id=app, user_id=user)
        print(f"  + {text}\n    -> {res['memory_id'][:8]} ({res['processing_time_ms']}ms)")

    section("第 2 轮写入：偏好变化 + 搬家（应触发演化链）")
    for text in [
        "其实我现在主要用 Python 做机器学习了，Java 基本不碰了。",
        "我从上海搬到北京了，入职了一家新公司。",
    ]:
        res = await client.add(content=text, app_id=app, user_id=user)
        print(f"  + {text}\n    -> {res['memory_id'][:8]} ({res['processing_time_ms']}ms)")

    section("检索：用户现在的编程语言偏好（期望命中 Python，并带 Java→Python 演化链）")
    out = await client.search(
        query="这位用户现在主要用什么编程语言？", app_ids=[app], user_id=user, limit=5, min_score=0.0
    )
    show_memories(out["memories"])

    section("检索：用户的基础画像 / 居住地（期望命中北京）")
    out = await client.search(
        query="用户是谁？住在哪里？做什么工作？",
        app_ids=[app],
        user_id=user,
        limit=5,
        min_score=0.0,
        profile_limit=5,
    )
    show_memories(out["memories"])

    section("全部记忆（ACTIVE）")
    for m in await client.list(app_id=app, user_id=user, limit=50):
        print(f"  - ({m['category']}) {m['content']}")


if __name__ == "__main__":
    asyncio.run(main())
