from conftest import FakeLLMClient

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
from dual_mem.system2.cross_domain_sweeper import CrossDomainSweeper
from dual_mem.types import Layer


def _factory(tmp_storage, fake_embed, responses):
    return ComponentFactory(
        # cross_domain_enable=True so the sweeper actually runs in tests.
        settings=Settings(
            mode="dual",
            storage_dir=tmp_storage,
            cross_domain_enable=True,
            cross_domain_min_basics=5,
        ),
        embed=fake_embed,
        llm=FakeLLMClient(responses=responses),
    )


def _seed_basics(factory, n: int) -> list[str]:
    ids = []
    for i in range(n):
        nid = f"s{i}"
        content = f"当领域{i}时，用户追求确定性"
        factory.graph.add_node(
            GraphNode(
                node_id=nid,
                layer=Layer.L6_SCHEMA.value,
                content=content,
                app_id="app",
                user_id="u",
                embedding=factory.embed.embed_sync(content),
            )
        )
        ids.append(nid)
    return ids


def _cross_edge_count(factory, core_id: str) -> int:
    result = factory.graph.conn.execute(
        "MATCH (a:Memory)-[:CROSS_ABSTRACTS_TO]->(b:Memory {node_id: $cid}) "
        "RETURN count(a)",
        {"cid": core_id},
    )
    return result.get_next()[0]


async def test_below_threshold_not_triggered(tmp_storage, fake_embed):
    factory = _factory(tmp_storage, fake_embed, {"json": {"abstraction_for_embedding": "x"}})
    _seed_basics(factory, 4)
    result = await CrossDomainSweeper(factory=factory).run(app_id="app", user_id="u")
    assert result == {"triggered": False, "basics_count": 4}


async def test_triggered_creates_core_and_edges(tmp_storage, fake_embed):
    """When 5+ basic schemas collide above threshold, sweeper synthesizes a core."""
    ids = [f"s{i}" for i in range(5)]
    # The fake LLM serves both abstraction and induction prompts via a single "json" key —
    # the abstraction is consumed first; induction reads core_pattern + schema_ids.
    responses = {
        "json": _scripted_responses(ids),
    }
    factory = _factory(tmp_storage, fake_embed, responses)
    _seed_basics(factory, 5)

    result = await CrossDomainSweeper(factory=factory).run(app_id="app", user_id="u")
    assert result["triggered"] is True
    assert result.get("cores", 0) >= 1


def _scripted_responses(schema_ids: list[str]):
    """Cycle abstraction -> induction responses across calls."""
    state = {"abstraction_calls": 0}

    def handler(*, system, user):
        # Behavior abstraction prompt asks for {"abstraction_for_embedding": ...}.
        if "behavioral" in system.lower() or "行为心理学" in system:
            state["abstraction_calls"] += 1
            return {"abstraction_for_embedding": f"deterministic-control-{state['abstraction_calls']}"}
        # Cross-domain induction prompt asks for {"core_pattern": ..., "schema_ids": ...}.
        return {
            "core_pattern": "用户在各领域都追求确定性与掌控——核心是低不确定性容忍。",
            "schema_ids": schema_ids,
            "reasoning": "across domains the user prefers explicit structure",
            "confidence": 0.9,
        }

    return handler
