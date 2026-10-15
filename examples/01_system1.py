"""Demo 1 — system1：JEV 写入版本链 + 多轮 messages。

每次 add 都是：保留 L0_RAW → 生成候选 → JEV 裁决 ADD/SKIP/DELETE 和
OVERRIDE/MERGE/SUPPLEMENT/TEMPORAL/NEGATE/CONFLICT → 追加新版本并链接历史。
system1 不跑 digest，因此没有 L3 跨证据推导。

环境要求：jev_*、llm_*、embed_* 都要有。会打真实 JEV、生成模型和 embedding。

运行：python examples/01_system1.py
期望：第二轮之后，「现在用什么编程语言」命中 Python，演化链里能看到 Java；
      「住在哪里」命中北京。list 默认只给 latest；include_history=True 能看到旧版本。
"""

import asyncio

from _common import example_settings, section, show_memories, show_write

from dual_mem import MemoryClient


async def main() -> None:
    client = MemoryClient(settings=example_settings("system1", "system1"))
    user = "bob"
    try:
        section("第 1 轮：单轮 content（画像 + 旧语言偏好）")
        for text in [
            "我叫王磊，今年 30 岁，是一名后端工程师，在上海工作。",
            "我最喜欢的编程语言是 Java，已经用了五年了，非常熟练。",
        ]:
            show_write(await client.add(content=text, user_id=user), text)

        section("第 2 轮：多轮 messages（语言换成 Python，居住地换成北京）")
        dialogue = [
            {
                "role": "user",
                "content": "其实我现在主要用 Python 做机器学习了，Java 基本不碰了。",
            },
            {"role": "assistant", "content": "好的，我帮你把语言偏好更新一下。"},
            {"role": "user", "content": "我从上海搬到北京了，入职了一家新公司。"},
        ]
        show_write(await client.add(messages=dialogue, user_id=user))

        section("检索：现在的编程语言（期望 Python，链上保留 Java）")
        out = await client.search(
            query="这位用户现在主要用什么编程语言？",
            user_id=user,
            limit=5,
            min_score=0.0,
        )
        show_memories(out.memories)

        section("检索：画像 / 居住地（期望 L2_PROFILE 命中北京）")
        out = await client.search(
            query="用户是谁？住在哪里？做什么工作？",
            user_id=user,
            limit=5,
            min_score=0.0,
            profile_limit=5,
        )
        show_memories(out.memories)

        section("当前版本（list 默认 latest）")
        for item in await client.list(user_id=user, limit=50):
            print(f"  - ({item.layer} {item.kind or item.category}) {item.content}")

        section("含历史版本（include_history=True，旧表述 is_latest=false）")
        for item in await client.list(user_id=user, limit=50, include_history=True):
            mark = "latest" if item.is_latest else "history"
            print(f"  - [{mark}] ({item.layer}) {item.content}")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
