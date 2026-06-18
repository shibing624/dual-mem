"""M3 端到端 smoke demo（从真实 SDK 用户视角）。

模拟一个 pro 模式的应用：用户先自我介绍 + 表达偏好，随后偏好发生改变。
用脚本化的 fake LLM / embed（无需 API key）走完整 System1 认知层：
extract(含 L0 工具) → reconcile(演化链) → summarize(L3)。

运行：python tmp/m3_smoke_demo.py
"""

import asyncio
import hashlib
import json
import math
import tempfile

from dual_mem import MemoryClient
from dual_mem.isolation import build_filter
from dual_mem.types import Layer, MemoryStatus


class DemoEmbed:
    def __init__(self, dim=64):
        self.dim = dim

    def embed(self, text):
        vec = []
        counter = 0
        while len(vec) < self.dim:
            for byte in hashlib.sha256(f"{text}:{counter}".encode()).digest():
                vec.append(byte / 255.0)
                if len(vec) >= self.dim:
                    break
            counter += 1
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class DemoLLM:
    """按 system prompt 类型 + user 内容路由返回脚本化结果。"""

    def chat_with_tools(self, *, system, user, tools, **kw):
        if "咖啡" in user:
            return {
                "content": json.dumps(
                    {
                        "identity": [{"content": "用户喜欢喝咖啡", "speculate": None, "tags": ["饮品"]}],
                        "facts": [{"content": "用户上周开始在家办公", "speculate": None, "tags": ["工作"]}],
                    },
                    ensure_ascii=False,
                ),
                "tool_calls": [
                    {"function": {"name": "update_basic_user_profile", "arguments": '{"name": "张三", "location": "北京"}'}}
                ],
            }
        return {
            "content": json.dumps(
                {"identity": [{"content": "用户现在更喜欢喝茶", "speculate": None, "tags": ["饮品"]}], "facts": []},
                ensure_ascii=False,
            ),
            "tool_calls": [],
        }

    def chat_json(self, *, system, user, **kw):
        if "搜索查询生成器" in system:
            return ["饮品偏好", "咖啡", "茶"]
        if "记忆管理系统" in system:
            # 第二轮：把"喝咖啡"的旧 identity 标为被取代
            return [
                {
                    "reason": "用户饮品偏好从咖啡变为茶",
                    "ops": [
                        {
                            "op": "ADD",
                            "content": "用户现在更喜欢喝茶",
                            "layer": "L4_IDENTITY",
                            "supersedes": [self.coffee_id],
                            "supersede_reason": "之前喜欢咖啡，现在改喝茶",
                            "tags": ["饮品"],
                        }
                    ],
                }
            ]
        return {"identity": [], "facts": []}

    def chat_text(self, *, system, user, **kw):
        return "用户张三在北京在家办公，饮品偏好从咖啡转向了茶。"


async def main():
    tmpdir = tempfile.mkdtemp(prefix="dual_mem_m3_")
    llm = DemoLLM()
    client = MemoryClient(mode="pro", storage_dir=tmpdir, embed=DemoEmbed(), llm=llm)

    app_id, user_id, agent_id = "demo_app", "user_zhangsan", "assistant"

    # 第 1 次写入：自我介绍 + 偏好（足够长以触发 L3 摘要）
    convo1 = (
        "我叫张三，住在北京。我特别喜欢喝咖啡，每天早上都要来一杯手冲。"
        "另外告诉你一件事，我上周开始在家办公了，感觉效率还挺高的。"
        "我们今天聊了很多关于生活和工作的话题，希望你能帮我记住这些。"
    ) * 8
    r1 = await client.add(content=convo1, app_id=app_id, user_id=user_id, agent_id=agent_id)
    print("== add #1 ==", r1["success"], "raw_id:", r1["memory_id"][:8])

    # 找到"喝咖啡"那条 L4，供第二次 reconcile 演化
    where_l4 = build_filter(
        app_ids=[app_id], user_id=user_id, layers=[Layer.L4_IDENTITY], statuses=[MemoryStatus.ACTIVE]
    )
    coffee_node = next(n for n in client.factory.vector.get_many(where_l4) if "咖啡" in n.content)
    llm.coffee_id = coffee_node.node_id

    # 第 2 次写入：偏好改变 → 应触发 supersedes 演化链
    r2 = await client.add(
        content="跟你说，我现在不太喝咖啡了，改成喝茶啦。", app_id=app_id, user_id=user_id, agent_id=agent_id
    )
    print("== add #2 ==", r2["success"], "raw_id:", r2["memory_id"][:8])

    # 展示各层结果
    print("\n--- 全部节点（按层）---")
    all_nodes = client.factory.vector.get_many(build_filter(app_ids=[app_id], user_id=user_id))
    for n in sorted(all_nodes, key=lambda x: x.layer.value):
        flag = "latest" if n.is_latest else "       "
        print(f"[{n.layer.value:14}] {n.status.value:10} {flag}  {n.content[:40]}")

    print("\n--- L0 basic profile 演化链 ---")
    where_l0 = build_filter(app_ids=[app_id], user_id=user_id, layers=[Layer.L0_BASIC_INFO])
    for n in sorted(client.factory.vector.get_many(where_l0), key=lambda x: x.gmt_created):
        print(f"  {n.status.value:10} is_latest={n.is_latest} kv={n.custom.get('basic_info_kv')}")

    print("\n--- 饮品偏好演化链（咖啡 → 茶）---")
    old_coffee = client.factory.vector.get(coffee_node.node_id)
    print(f"  旧: status={old_coffee.status.value} is_latest={old_coffee.is_latest} superseded_by={[i[:8] for i in old_coffee.superseded_by]}")
    tea = next(n for n in client.factory.vector.get_many(where_l4) if "茶" in n.content)
    print(f"  新: status={tea.status.value} is_latest={tea.is_latest} supersedes={[i[:8] for i in tea.supersedes]}")

    summaries = client.factory.vector.get_many(
        build_filter(app_ids=[app_id], user_id=user_id, layers=[Layer.L3_SUMMARY])
    )

    assert old_coffee.status is MemoryStatus.SUPERSEDED and not old_coffee.is_latest
    assert tea.supersedes == [coffee_node.node_id]
    assert len(summaries) == 1, "应产出 1 条 L3 摘要"
    print(f"\nL3 摘要: {summaries[0].content}")
    print("OK: 演化链正确，L0/L2/L3/L4 全部就位。")


if __name__ == "__main__":
    asyncio.run(main())
