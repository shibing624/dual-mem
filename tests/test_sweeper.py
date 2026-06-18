from conftest import FakeLLMClient

from dual_mem.config import Settings
from dual_mem.registry import ComponentFactory
from dual_mem.storage.graph_store import GraphNode
from dual_mem.system2.cross_domain_sweeper import CrossDomainSweeper
from dual_mem.types import Layer


def _factory(tmp_storage, fake_embed, responses):
    return ComponentFactory(
        settings=Settings(mode="ultra", storage_dir=tmp_storage),
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
                embedding=factory.embed.embed(content),
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


def test_below_threshold_not_triggered(tmp_storage, fake_embed):
    factory = _factory(tmp_storage, fake_embed, {"json": {}})
    _seed_basics(factory, 4)
    result = CrossDomainSweeper(factory=factory).run(app_id="app", user_id="u")
    assert result == {"triggered": False, "basics_count": 4}


def test_triggered_creates_core_and_edges(tmp_storage, fake_embed):
    ids = [f"s{i}" for i in range(5)]
    factory = _factory(
        tmp_storage,
        fake_embed,
        {
            "json": {
                "content": "用户在各领域都追求确定性与掌控——核心是低不确定性容忍。",
                "schema_ids": ids,
            }
        },
    )
    _seed_basics(factory, 5)

    result = CrossDomainSweeper(factory=factory).run(app_id="app", user_id="u")
    assert result["triggered"] is True
    assert result["abstracted"] == 5

    core_id = result["core_id"]
    schemas = factory.graph.list_by_layer(
        layer=Layer.L6_SCHEMA.value, user_id="u", app_ids=["app"]
    )
    core = next(n for n in schemas if n.node_id == core_id)
    assert (core.custom or {}).get("sub_type") == "core"
    assert _cross_edge_count(factory, core_id) == 5
