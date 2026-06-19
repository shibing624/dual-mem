"""Demo 5 — dual + system2_trigger_mode=scheduled：定时批量 System2 蒸馏。

与 02_dual.py 的手动 digest() / per_write 不同，scheduled 模式在首次 write 后启动
后台 loop，每隔 system2_schedule_interval_sec 秒自动 drain 队列（类似「周期性巩固记忆」）。

环境要求：llm_* + embed_*；会产生真实 LLM 调用。

运行：python examples/05_scheduled_system2.py
期望：sleep 一个 interval 后 search 能命中 L6 Schema；结束时必须 aclose() 停掉 loop。
"""
import asyncio

from _common import fresh_storage, section, show_memories

from dual_mem import MemoryClient
from dual_mem.config import Settings

# demo 用短间隔；生产环境 config.yaml 默认 300 秒
SCHEDULE_SEC = 15


async def main() -> None:
    settings = Settings(
        mode="dual",
        storage_dir=fresh_storage("scheduled"),
        system2_trigger_mode="scheduled",
        system2_schedule_interval_sec=SCHEDULE_SEC,
    )
    client = MemoryClient(settings=settings)
    app, user = "default", "dave"
    try:
        section(f"写入事实（仅入队，不立即 digest；{SCHEDULE_SEC}s 后由 scheduled loop 处理）")
        facts = [
            "我上周把整个衣柜按颜色和季节重新分类整理了一遍。",
            "我出差前一定会列一张详细的清单，逐项打勾确认才安心。",
            "我的代码仓库里每个文件夹都有规范命名和一份 README。",
            "我记账精确到每一笔几块钱的小额支出。",
            "下个月我要参加一场马拉松比赛，正在按周制定训练计划。",
        ]
        for text in facts:
            res = await client.add(content=text, app_id=app, user_id=user)
            print(f"  + {text}\n    -> id={res.memory_id[:8]}")

        section(f"等待 scheduled loop（{SCHEDULE_SEC + 2}s）…")
        await asyncio.sleep(SCHEDULE_SEC + 2)

        section("检索：行为模式（期望 profile 出现 L6 Schema）")
        out = await client.search(
            query="这位用户在做事方式上有什么稳定的行为模式？",
            app_ids=[app],
            user_id=user,
            limit=5,
            min_score=0.0,
            profile_limit=5,
        )
        show_memories(out.memories)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
