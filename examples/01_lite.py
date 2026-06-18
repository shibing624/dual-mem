"""Demo 1 — lite 档：纯向量召回，零 LLM。

场景：一个客服 Agent 把用户零散偏好直接存进记忆，之后用自然语言检索。
lite 档只调用 embedding（无 LLM 抽取/整理），写入即 L1_RAW，召回走 normal 路。

运行：python examples/01_lite.py
"""
import asyncio

from _common import fresh_storage, section, show_memories

from dual_mem import MemoryClient


async def main() -> None:
    client = MemoryClient(mode="lite", storage_dir=fresh_storage("lite"))
    section("lite 写入（仅 embedding，无 LLM）")
    notes = [
        "用户喜欢喝美式咖啡，不加糖",
        "用户对花生过敏",
        "用户偏好靠窗的座位",
        "用户常用的快递地址是公司前台",
    ]
    for n in notes:
        res = await client.add(content=n, app_id="default", user_id="alice")
        print(f"  + {n}  -> {res['memory_id'][:8]}  ({res['processing_time_ms']}ms)")

    section("检索：用户的饮品偏好")
    out = await client.search(
        query="这位用户喝什么咖啡？", app_ids=["default"], user_id="alice", limit=3, min_score=0.0
    )
    show_memories(out["memories"])

    section("检索：饮食禁忌")
    out = await client.search(
        query="用户有什么过敏或忌口？", app_ids=["default"], user_id="alice", limit=3, min_score=0.0
    )
    show_memories(out["memories"])


if __name__ == "__main__":
    asyncio.run(main())
