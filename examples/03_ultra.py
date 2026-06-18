"""Demo 3 — ultra 档：System2 异步认知（Schema / Intention）。

场景：用户在某领域留下多条同质事实，System2 在 digest 时聚类并沉淀出高层认知：
- L6 Schema：跨多条证据归纳出的行为模式（"当…时，用户…——反映…"）。
- L7 Intention：用户表达的具体未来计划。
ultra 档 = pro 全部能力 + System2 异步加工 + 图库。

运行：python examples/03_ultra.py
"""
import asyncio

from _common import fresh_storage, section, show_memories

from dual_mem import MemoryClient


async def main() -> None:
    client = MemoryClient(mode="ultra", storage_dir=fresh_storage("ultra"))
    app, user = "default", "carol"

    # 给一组「互不取代但同主题」的离散事实：System2 才能聚类并归纳出行为模式。
    # （若给的是会被 Reconciler 合并成同一条 identity 的近义句，聚类样本会不足。）
    # 注意：这里刻意用串行写入——System2 聚类依赖这些离散事实各自落库；
    # 若改并发（asyncio.gather），重叠的 reconcile 会把同主题事实合并成一条，聚类样本不足，
    # 反而出不了 Schema。并发写入只适合彼此完全独立、无需跨事实归纳的场景（见 examples/README）。
    section("写入同领域多条离散事实（System1 先落 L2/L4）")
    facts = [
        "我上周把整个衣柜按颜色和季节重新分类整理了一遍。",
        "我出差前一定会列一张详细的清单，逐项打勾确认才安心。",
        "我的代码仓库里每个文件夹都有规范命名和一份 README。",
        "我记账精确到每一笔几块钱的小额支出。",
        "我家的调料瓶都贴了标签并按使用频率排列。",
        "下个月我要参加一场马拉松比赛，正在按周制定训练计划。",
    ]
    for f in facts:
        res = await client.add(content=f, app_id=app, user_id=user)
        print(f"  + {f}  -> {res['memory_id'][:8]}")

    section("digest()：触发 System2 聚类 + 出图（Schema/Intention）")
    d = await client.digest()
    print(f"  processed={d['processed']}  cores_created={d.get('cores_created')}")

    section("检索：用户的行为模式（期望 profile 召回 L6 Schema）")
    out = await client.search(
        query="这位用户在做事方式上有什么稳定的行为模式？",
        app_ids=[app],
        user_id=user,
        limit=5,
        min_score=0.0,
        profile_limit=5,
        intention_limit=3,
    )
    show_memories(out["memories"])


if __name__ == "__main__":
    asyncio.run(main())
