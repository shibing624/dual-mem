"""Demo 5 — SDK 里 01/02 没覆盖、且结果不取决于 JEV 选哪种 update_type 的路径。

覆盖：add_raw 只写 L0 → distill 抽取且不新增原文 → 默认检索不含 L0、
search_conversation 只查 L0 → update 返回新 id 并接上演化链 → delete 归档，
旧 id 再删是 409，get 仍能读到归档行。system1 的 digest() 直接返回 processed=0。

运行：python examples/05_lifecycle.py
失败会非零退出。
"""
import asyncio
import sys

from _common import example_settings, section, show_write

from dual_mem import MemoryClient


def _fail(message: str) -> None:
    print("FAIL:", message)
    raise SystemExit(1)


async def main() -> None:
    client = MemoryClient(settings=example_settings("system1", "lifecycle"))
    user = "frank"
    try:
        section("add_raw：只写 L0，不调用抽取")
        first = await client.add_raw(content="我每周三晚上去游泳。", user_id=user)
        second = await client.add_raw(
            content="教练说我的自由泳换气还不稳定。", user_id=user
        )
        show_write(first, "我每周三晚上去游泳。")
        show_write(second, "教练说我的自由泳换气还不稳定。")
        if first.extracted_count != 0 or second.extracted_count != 0:
            _fail("add_raw 不应产生结构化版本")
        for raw_id in (first.memory_id, second.memory_id):
            item = await client.get(raw_id)
            if item is None or item.layer != "L0_RAW":
                _fail(f"get({raw_id[:8]}) 不是 L0_RAW")

        section("distill：沿用已有 L0，不新增原文")
        distilled = await client.distill(
            messages=[
                {"role": "user", "content": "我每周三晚上去游泳。"},
                {"role": "assistant", "content": "记下了。"},
                {"role": "user", "content": "教练说我的自由泳换气还不稳定。"},
            ],
            user_id=user,
            source_node_ids=[first.memory_id, second.memory_id],
        )
        show_write(distilled)
        if distilled.extracted_count < 1 or not distilled.extra_node_ids:
            _fail("distill 没有写出结构化版本")
        raws = [
            item
            for item in await client.list(user_id=user, limit=50)
            if item.layer == "L0_RAW"
        ]
        if len(raws) != 2:
            _fail(f"distill 之后 L0 应为 2 条，实际 {len(raws)}")
        structured_id = distilled.extra_node_ids[0]
        structured = await client.get(structured_id)
        if structured is None or structured.source_node_id not in {
            first.memory_id,
            second.memory_id,
        }:
            _fail("结构化版本没有指回 L0 source_node_id")
        print(
            f"  source={structured.source_node_id[:8]}  "
            f"layer={structured.layer}  {structured.content}"
        )

        section("默认检索不含 L0；search_conversation 只查原文")
        qa = await client.search(
            query="每周三游泳",
            user_id=user,
            min_score=0.0,
        )
        qa_ids = {item.memory_id for item in qa.memories.flatten()}
        if first.memory_id in qa_ids or second.memory_id in qa_ids:
            _fail("默认 search 返回了 L0 原文")
        conv = await client.search_conversation(
            query="自由泳换气",
            user_id=user,
            min_score=0.0,
        )
        conv_items = conv.memories.normal
        conv_ids = {item.memory_id for item in conv_items}
        if second.memory_id not in conv_ids:
            _fail("search_conversation 没有命中第二条原文")
        if any(item.layer != "L0_RAW" for item in conv_items):
            _fail("search_conversation 混入了非 L0")
        print(f"  qa={len(qa_ids)} 条结构化  conversation={len(conv_items)} 条原文")

        section("update：追加 OVERRIDE，返回新 id，旧 id 仍可 get")
        updated = await client.update(structured_id, "用户每周三晚上去游泳，换气已经稳定。")
        if not updated.success or not updated.memory_id:
            _fail(f"update 失败 error_code={updated.error_code}")
        if updated.memory_id == structured_id:
            _fail("update 应返回新 id")
        old = await client.get(structured_id)
        new = await client.get(updated.memory_id)
        if old is None or new is None or old.is_latest or not new.is_latest:
            _fail("update 后旧版本应 is_latest=false，新版本为 latest")
        print(f"  new={updated.memory_id[:8]}  old={structured_id[:8]} is_latest={old.is_latest}")

        found = await client.search(
            query="换气已经稳定",
            user_id=user,
            min_score=0.0,
            limit=5,
            profile_limit=5,
            include_derived=True,
        )
        hit = next(
            (item for item in found.memories.flatten() if item.memory_id == updated.memory_id),
            None,
        )
        if hit is None or not hit.evolution_chain or len(hit.evolution_chain) < 2:
            _fail("检索没有带回新旧两版演化链")
        print(
            "  演化链: "
            + " → ".join(f"{node.layer}:{node.content[:20]}" for node in hit.evolution_chain)
        )

        section("delete：归档当前版本；再删旧 id 是 409；get 仍能读归档行")
        deleted = await client.delete(updated.memory_id)
        if not deleted.success:
            _fail("delete 当前版本失败")
        again = await client.delete(structured_id)
        if again.success or again.error_code != 409:
            _fail(f"删除非 latest 应是 409，实际 success={again.success} code={again.error_code}")
        archived = await client.get(updated.memory_id)
        if archived is None or archived.is_latest:
            _fail("归档后 get 应仍返回该行，且 is_latest=false")
        latest_ids = {
            item.memory_id
            for item in await client.list(user_id=user, limit=50)
        }
        if updated.memory_id in latest_ids or structured_id in latest_ids:
            _fail("list 默认不应再给出已归档的版本链")
        print(f"  archived={updated.memory_id[:8]} is_latest={archived.is_latest}")

        section("system1 的 digest 不生成 L3")
        digest = await client.digest()
        if digest.processed != 0:
            _fail(f"system1 digest 应 processed=0，实际 {digest.processed}")
        print(f"  processed={digest.processed}")

        scopes = await client.list_scopes()
        if not any(scope.user_id == user for scope in scopes):
            _fail("list_scopes 没有这个用户")
        print("  scopes:", [(scope.user_id, scope.memory_count) for scope in scopes])
        print("\nOK")
    finally:
        await client.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except Exception as exc:
        print("FAIL:", type(exc).__name__, exc)
        sys.exit(1)
