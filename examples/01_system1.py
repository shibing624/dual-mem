"""Demo 1 — system1 档：System1 抽取 + 演化链 + 多轮 messages 输入。

场景：用户偏好随时间变化，记忆系统应自动用新认知"取代"旧认知，并保留可追溯的演化链。
system1 档每次 `add` 会跑 LLM：Gate → Extractor → fast-write 直落 L2/L4 → 异步 reconcile
合并演化链（默认 `reconcile_sync=false`；置 true 则写路径同步 reconcile）。本 demo 同时
演示 `add(messages=[...])`：多轮对话写入时 Gate 取各轮 user 文本 novelty 的最大值。

环境要求：`llm_*` + `embed_*` 全部齐全；会产生真实 LLM 调用与少量费用。

运行：python examples/01_system1.py
期望：第二轮写入后，"现在用什么编程语言"应命中 Python，并附 Java→Python 演化链；
      "住在哪里"应命中北京。
"""
import asyncio

from _common import fresh_storage, section, show_memories

from dual_mem import MemoryClient


async def main() -> None:
    client = MemoryClient(mode="system1", storage_dir=fresh_storage("system1"))
    user = "bob"
    try:
        section("第 1 轮：单轮 content 写入（自我介绍 + 旧偏好）")
        for text in [
            "我叫王磊，今年 30 岁，是一名后端工程师，在上海工作。",
            "我最喜欢的编程语言是 Java，已经用了五年了，非常熟练。",
        ]:
            res = await client.add(content=text, user_id=user)
            print(
                f"  + {text}\n    -> id={res.memory_id[:8]}  "
                f"gate={res.gate_passed}/{res.gate_score}  "
                f"({res.processing_time_ms / 1000:.2f}s)"
            )

        section("第 2 轮：多轮 messages 写入（偏好变化 + 搬家，触发演化链）")
        # 助手寒暄不会贡献 novelty；novelty 取所有 user 轮的 max。
        dialogue = [
            {"role": "user", "content": "其实我现在主要用 Python 做机器学习了，Java 基本不碰了。"},
            {"role": "assistant", "content": "好的，我帮你把语言偏好更新一下。"},
            {"role": "user", "content": "我从上海搬到北京了，入职了一家新公司。"},
        ]
        res = await client.add(messages=dialogue, user_id=user)
        print(
            f"  -> id={res.memory_id[:8]}  extracted={res.extracted_count}  "
            f"gate={res.gate_passed}/{res.gate_score}  "
            f"({res.processing_time_ms / 1000:.2f}s)"
        )

        section("检索：现在的编程语言（期望 Python，并带 Java→Python 演化链）")
        out = await client.search(
            query="这位用户现在主要用什么编程语言？",
            user_id=user, limit=5, min_score=0.0,
        )
        show_memories(out.memories)

        section("检索：基础画像 / 居住地（期望 profile 命中北京）")
        out = await client.search(
            query="用户是谁？住在哪里？做什么工作？",
            user_id=user, limit=5, min_score=0.0, profile_limit=5,
        )
        show_memories(out.memories)

        section("全部 ACTIVE 记忆")
        for item in await client.list(user_id=user, limit=50):
            print(f"  - ({item.category}) {item.content}")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
