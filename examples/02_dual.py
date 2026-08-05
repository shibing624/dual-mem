"""Demo 2 — dual 档：显式 System2 认知（Schema / Intention）。

场景：用户在某领域留下多条同质事实，System2 在 `digest()` 时聚类并沉淀出高层认知：
  - L6 Schema：跨多条证据归纳出的行为模式（"当…时，用户…——反映…"）。
  - L7 Intention：用户表达的具体未来计划。

dual = system1 全部能力 + System2 ReAct（多轮 function-calling，受 `system2_max_iters` 限制）+
图库（Kuzu）。

环境要求：`llm_*` + `embed_*`；会产生数次真实 LLM 调用（Reader/Writer/System2）。

运行：python examples/02_dual.py
期望：digest 后能看到至少一条 L6 schema 出现在 profile 召回。
"""
import asyncio

from _common import fresh_storage, section, show_memories

from dual_mem import MemoryClient
from dual_mem.config import Settings


async def main() -> None:
    settings = Settings(
        mode="dual",
        storage_dir=fresh_storage("dual"),
    )
    client = MemoryClient(settings=settings)
    user = "carol"
    try:
        # 给一组「互不取代但同主题」的离散事实：System2 才能聚类并归纳出行为模式。
        # 注意：这里刻意串行写入——并发会让 reconcile 把同主题事实合并成一条，
        # 聚类样本不足，反而出不了 Schema。
        section("写入同领域多条离散事实（System1 先落 L2/L4）")
        facts = [
            "我上周把整个衣柜按颜色和季节重新分类整理了一遍。",
            "我出差前一定会列一张详细的清单，逐项打勾确认才安心。",
            "我的代码仓库里每个文件夹都有规范命名和一份 README。",
            "我记账精确到每一笔几块钱的小额支出。",
            "我家的调料瓶都贴了标签并按使用频率排列。",
            "下个月我要参加一场马拉松比赛，正在按周制定训练计划。",
        ]
        for text in facts:
            res = await client.add(content=text, user_id=user)
            print(f"  + {text}\n    -> id={res.memory_id[:8]}")

        section("digest()：显式触发 reconcile + System2 ReAct")
        digest = await client.digest()
        print(f"  processed={digest.processed}")

        section("检索：行为模式（期望 profile 召回 L6 Schema）")
        out = await client.search(
            query="这位用户在做事方式上有什么稳定的行为模式？",
            user_id=user,
            limit=5,
            min_score=0.0,
            profile_limit=5,
            intention_limit=3,  # 打开 L7 Intention 召回（默认关闭）
        )
        show_memories(out.memories)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
