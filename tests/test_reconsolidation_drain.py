"""R3: Reconsolidation drain 实现 — 入队的 reconsolidation task 被真正消费。

测试目标：
1. dual search → reconsolidation 入队 → _digest_reconsolidation_pending 消费 → 节点 custom 标记 reactivation
2. 显著情绪差时打 reactivation flag；平淡 query 不打
3. scheduled 模式下 reconsolidation 也走 _run_reconsolidation 而不是只 log
4. per_write 下 client.search 自动 fire-and-forget drain（at-least 不报错）
"""
from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


def _seed_emotional_node(client, node_id: str, content: str, *, arousal: float = 0.0) -> None:
    custom = {"emotional_arousal": arousal} if arousal else None
    n = MemoryNode(
        content=content, layer=Layer.L4_IDENTITY,
        app_id="app", user_id="u", status=MemoryStatus.ACTIVE,
        node_id=node_id, custom=custom,
    )
    n.embedding = client.factory.embed.embed_sync(content)
    client.factory.vector.upsert([n])


async def test_reconsolidation_drain_marks_reactivation_for_emotional_query(tmp_storage, fake_embed):
    """High-arousal query against a calm stored memory → reactivation flag set."""
    settings = Settings(mode="dual", storage_dir=tmp_storage,
                        system2_trigger_mode="manual", gate_enabled=False)
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={}))

    _seed_emotional_node(client, "n1", "用户喜欢登山", arousal=0.0)

    # 直接入一条 reconsolidation task（绕过 search hook，独立测 drain 行为）
    client.factory.cache.enqueue_s2_task(
        user_id="u", app_id="app", agent_id="",
        task_type="reconsolidation",
        payload={"query": "我崩溃了！太焦虑了！", "node_ids": ["n1"]},
    )

    # 跑 drain
    processed = await client.writer._digest_reconsolidation_pending()  # type: ignore[attr-defined]
    assert processed == 1

    n = client.factory.vector.get("n1")
    assert n.custom is not None
    assert n.custom.get("reactivation") is True
    assert "reactivation_at" in n.custom
    assert "last_reactivated_at" in n.custom

    await client.aclose()


async def test_reconsolidation_drain_calm_query_only_bumps_timestamp(tmp_storage, fake_embed):
    """Low-arousal query → only last_reactivated_at, no reactivation flag."""
    settings = Settings(mode="dual", storage_dir=tmp_storage,
                        system2_trigger_mode="manual", gate_enabled=False)
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={}))
    _seed_emotional_node(client, "n2", "用户喜欢喝咖啡", arousal=0.0)

    client.factory.cache.enqueue_s2_task(
        user_id="u", app_id="app", agent_id="",
        task_type="reconsolidation",
        payload={"query": "用户的偏好是什么", "node_ids": ["n2"]},
    )
    await client.writer._digest_reconsolidation_pending()  # type: ignore[attr-defined]

    n = client.factory.vector.get("n2")
    assert n.custom is not None
    assert "last_reactivated_at" in n.custom
    assert n.custom.get("reactivation") is None  # 平淡 query 不打 flag

    await client.aclose()


async def test_reconsolidation_drain_logs_pipeline_stage(tmp_storage, fake_embed):
    """Drain 应该写 RECONSOLIDATION_DRAIN pipeline log 包含 flagged_reactivation 计数。"""
    settings = Settings(mode="dual", storage_dir=tmp_storage,
                        system2_trigger_mode="manual", gate_enabled=False)
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={}))
    _seed_emotional_node(client, "n3", "memo")

    client.factory.cache.enqueue_s2_task(
        user_id="u", app_id="app", agent_id="",
        task_type="reconsolidation",
        payload={"query": "焦虑！崩溃了！", "node_ids": ["n3"]},
    )
    await client.writer._digest_reconsolidation_pending()  # type: ignore[attr-defined]

    # log_pipeline 用了 request_id="reconsolidation::u" 模式
    rows = client.factory.cache.list_pipeline_logs("reconsolidation::u")
    assert any(r["stage"] == "RECONSOLIDATION_DRAIN" for r in rows)
    drain_log = next(r for r in rows if r["stage"] == "RECONSOLIDATION_DRAIN")
    assert "n_nodes" in drain_log["payload"]
    assert "flagged_reactivation" in drain_log["payload"]

    await client.aclose()


async def test__digest_pending_runs_reconsolidation(tmp_storage, fake_embed):
    """digest() 调用的 _digest_pending 应该处理 cognition 和 reconsolidation 两类任务。"""
    settings = Settings(mode="dual", storage_dir=tmp_storage,
                        system2_trigger_mode="manual", gate_enabled=False)
    client = MemoryClient(settings=settings, embed=fake_embed,
                          llm=FakeLLMClient(responses={"tools": [{"content": "", "tool_calls": []}]}))
    _seed_emotional_node(client, "n4", "memo")
    client.factory.cache.enqueue_s2_task(
        user_id="u", app_id="app", agent_id="",
        task_type="reconsolidation",
        payload={"query": "崩溃焦虑！", "node_ids": ["n4"]},
    )
    digest = await client.digest()
    assert digest.processed >= 1

    n = client.factory.vector.get("n4")
    # reconsolidation 已被处理 → last_reactivated_at 应该存在
    assert n.custom is not None and "last_reactivated_at" in n.custom

    await client.aclose()
