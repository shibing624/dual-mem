"""Demo 2 — dual：每次 add 先完成和 system1 相同的 JEV 写入，digest() 再写 L3。

digest 不重跑 reconcile。它只把尚未覆盖的 L1/L2 证据做成 L3_DERIVED
（跨证据模式、摘要）。意图仍是 L1 的 kind=intention，要 intention_limit>0 才进 proactive。

环境要求：jev_*、llm_*、embed_*。digest 会再打一次生成模型和 JEV。

运行：python examples/02_dual.py
期望：digest.processed >= 1，profile 里出现 L3；proactive 里能看到马拉松计划。
"""
import asyncio

from _common import example_settings, section, show_memories, show_write

from dual_mem import MemoryClient


async def main() -> None:
    client = MemoryClient(settings=example_settings("dual", "dual"))
    user = "carol"
    try:
        # 这些事实彼此不是同一维度的状态替换，JEV 应各自 SUPPLEMENT。
        # 串行写是为了让「整理习惯」和「马拉松计划」按叙述顺序落库；
        # 同 user 的并发 add 默认不拿写锁，靠版本 CAS 重试。
        section("写入同领域多条事实（每次 add 已完成 JEV，不是先堆原文再 digest）")
        facts = [
            "我上周把整个衣柜按颜色和季节重新分类整理了一遍。",
            "我出差前一定会列一张详细的清单，逐项打勾确认才安心。",
            "我的代码仓库里每个文件夹都有规范命名和一份 README。",
            "我记账精确到每一笔几块钱的小额支出。",
            "我家的调料瓶都贴了标签并按使用频率排列。",
            "下个月我要参加一场马拉松比赛，正在按周制定训练计划。",
        ]
        for text in facts:
            show_write(await client.add(content=text, user_id=user), text)

        section("digest()：只生成 L3，不重新裁决已写入的 L1/L2")
        digest = await client.digest()
        print(f"  processed={digest.processed}")

        section("检索：行为模式进 profile（L3），马拉松计划进 proactive")
        out = await client.search(
            query="这位用户在做事方式上有什么稳定的行为模式？接下来有什么计划？",
            user_id=user,
            limit=5,
            min_score=0.0,
            profile_limit=5,
            intention_limit=3,
        )
        show_memories(out.memories)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
