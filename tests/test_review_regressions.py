"""Regression tests for the review of the decision-protocol change.

1. Two HISTORICAL reports in one batch may attach behind the same current head.
2. A crash after the version plan commits cannot lose its review intents.
3. ``add(request_id=...)`` on the tool-message (coding) path is idempotent.
4. An older source cannot MERGE into (and so retire) a newer head.
5. A crash after a review re-extraction commits replays instead of re-deciding.
"""

import pytest

from conftest import FakeJev, FakeLLMClient
from dual_mem import IdempotencyConflictError, MemoryClient
from dual_mem.agent.mem_agent import MemAgent
from dual_mem.agent.reconciler import _action_question
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode


def echo_lines(system, user):
    text = user.split("Conversation:\n", 1)[-1].strip()
    return {"facts": [{"content": line} for line in text.splitlines() if line]}


def make(tmp_storage, embed, jev, *, extract=echo_lines, **kw):
    return MemoryClient(
        settings=Settings(storage_dir=tmp_storage, **kw),
        embed=embed,
        llm=FakeLLMClient({"extract": extract}),
        jev=jev,
    )


class ProcessDied(Exception):
    """Stands in for a process crash at a precise point (SystemExit escapes asyncio)."""


def seed_head(c, embed, *, memory_at: int | None = 2_000):
    head = MemoryNode(
        node_id="head",
        content="Lives in Shanghai",
        layer=Layer.L2_PROFILE,
        kind="profile",
        app_id="default",
        user_id="u",
        memory_at=memory_at,
        embedding=embed.embed_sync("Lives in Shanghai"),
    )
    c.factory.vector.upsert([head])
    return head


# --- 1. HISTORICAL x2 on one head ------------------------------------------------


async def test_two_historical_reports_attach_behind_the_same_head(
    tmp_storage, fake_embed
):
    def script(state, questions):
        out = {}
        for name, q in questions.items():
            if name.endswith(":target"):
                out[name] = "head"
            elif name.endswith(":action"):
                assert "ADD_HISTORICAL" in q["criteria"]
                out[name] = "ADD_HISTORICAL"
        return out

    c = make(tmp_storage, fake_embed, FakeJev(script))
    seed_head(c, fake_embed)
    result = await c.add(
        content="Lived in Beijing\nLived in Wuhan", user_id="u", memory_at=1_000
    )
    assert result.status == "committed"
    assert [d.update_type for d in result.decisions] == ["HISTORICAL", "HISTORICAL"]
    head = c.factory.vector.get("head")
    assert head.is_latest
    assert set(head.supersedes) == set(result.extra_node_ids)
    for nid in result.extra_node_ids:
        old = c.factory.vector.get(nid)
        assert not old.is_latest and old.superseded_by == ["head"]
    await c.aclose()


async def test_retiring_and_historical_on_one_head_still_replans(
    tmp_storage, fake_embed
):
    """A head that is retired by one op cannot also take a HISTORICAL in the batch."""
    c = make(tmp_storage, fake_embed, FakeJev())
    seed_head(c, fake_embed)
    agent = MemAgent(factory=c.factory)
    from dual_mem.agent.reconciler import ReconcileOp
    from dual_mem.errors import DecisionPlanError

    raw = MemoryNode(
        content="x", layer=Layer.L0_RAW, app_id="default", user_id="u", node_id="raw"
    )
    head_rev = c.factory.vector.get("head").revision
    ops = [
        ReconcileOp(
            op="ADD", content="Lives in Hangzhou", update_type="OVERRIDE",
            supersedes=["head"], target_ids=["head"], proposal_ids=["p0"],
            expected_revisions={"head": head_rev}, embedding=fake_embed.embed_sync("a"),
        ),
        ReconcileOp(
            op="ADD", content="Lived in Wuhan", update_type="HISTORICAL",
            target_ids=["head"], proposal_ids=["p1"],
            expected_revisions={"head": head_rev}, embedding=fake_embed.embed_sync("b"),
        ),
    ]
    with pytest.raises(DecisionPlanError, match="consume one target"):
        await agent._prepare_nodes(ops, raw, "r")
    await c.aclose()


# --- 4. older source cannot MERGE ------------------------------------------------


def test_older_source_action_set_excludes_every_retiring_action():
    criteria = _action_question("p0", has_targets=True, older_than_target=True)[
        "criteria"
    ]
    assert set(criteria) == {
        "ADD_HISTORICAL",
        "ADD_CONFLICT",
        "SKIP",
        "INVALID",
        "NEED_CONTEXT",
    }
    instructions = _action_question("p0", has_targets=True, older_than_target=True)[
        "instructions"
    ]
    assert "ADD_MERGE when" not in instructions and "use MERGE" not in instructions


async def test_older_source_merge_answer_is_rejected_not_applied(
    tmp_storage, fake_embed
):
    jev = FakeJev({"p0:target": "head", "p0:action": "ADD_MERGE"})
    c = make(tmp_storage, fake_embed, jev)
    seed_head(c, fake_embed)
    from dual_mem import DecisionContextError

    with pytest.raises(DecisionContextError, match="unsupported answer"):
        await c.add(content="Lived in Beijing", user_id="u", memory_at=1_000)
    assert c.factory.vector.get("head").is_latest
    await c.aclose()


# --- 2. crash between commit and review queue ------------------------------------


def need_context_script(state, questions):
    out = {}
    for name in questions:
        if name.endswith(":target"):
            out[name] = "head"
        elif name.endswith(":action"):
            out[name] = "NEED_CONTEXT"
    return out


def tool_messages(solution="Use BEGIN IMMEDIATE to validate before commit"):
    return [
        {"role": "user", "content": "Fix the SDK transaction"},
        {"role": "tool", "content": solution, "tool_call_id": "tx"},
    ]


def coding_client(tmp_storage, embed, jev=None):
    return MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=embed,
        llm=FakeLLMClient(
            {
                "json": {
                    "candidates": [
                        {
                            "task": "How to publish transactions",
                            "solution": "Use BEGIN IMMEDIATE to validate before commit",
                            "search_keys": [],
                        }
                    ]
                }
            }
        ),
        jev=jev or FakeJev({"scene": "CODING", "0:action": "ADD_SUPPLEMENT"}),
    )


def procedure_nodes(c):
    return c.factory.vector.get_many({"kind": "procedure"})


def test_requests_index_is_backfilled_for_existing_journals(tmp_storage, fake_embed):
    import sqlite3

    from dual_mem.storage.vector_store import ChromaVectorStore

    store = ChromaVectorStore(tmp_storage)
    node = MemoryNode(
        node_id="n", content="x", layer=Layer.L1_FACT, app_id="a", user_id="u",
        embedding=fake_embed.embed_sync("x"),
    )
    assert store.apply_version_plan(
        nodes=[node], expected={}, archives=[], scope=("a", "u", ""), request_key="k"
    )
    store.close()
    journal = sqlite3.connect(f"{tmp_storage}/versions.db")
    journal.execute("DROP TABLE requests")
    journal.commit()
    journal.close()

    reopened = ChromaVectorStore(tmp_storage)
    assert reopened.completed_request("k")["extra_node_ids"] == ["n"]
    assert reopened.completed_request("missing") is None
    reopened.close()


async def test_tool_path_retry_after_failed_publish_reuses_ids(
    tmp_storage, fake_embed, monkeypatch
):
    c = coding_client(tmp_storage, fake_embed)
    vector = c.factory.vector
    original = vector.apply_version_plan

    def die(**kwargs):
        if kwargs.get("request_key", "").startswith("procedure:"):
            raise ProcessDied("died before publish")
        return original(**kwargs)

    monkeypatch.setattr(vector, "apply_version_plan", die)
    with pytest.raises(ProcessDied):
        await c.add(messages=tool_messages(), user_id="u", request_id="t1")
    assert procedure_nodes(c) == []
    monkeypatch.setattr(vector, "apply_version_plan", original)

    result = await c.add(messages=tool_messages(), user_id="u", request_id="t1")
    again = await c.add(messages=tool_messages(), user_id="u", request_id="t1")
    assert [n.node_id for n in procedure_nodes(c)] == result.procedure_ids
    assert again.procedure_ids == result.procedure_ids
    await c.aclose()


async def test_tool_path_without_request_id_is_never_replayed(tmp_storage, fake_embed):
    """A generated request_id can never be replayed: each write is its own version."""
    c = coding_client(tmp_storage, fake_embed)
    first = await c.add(messages=tool_messages(), user_id="u")
    second = await c.add(messages=tool_messages(), user_id="u")
    assert first.memory_id != second.memory_id
    assert first.procedure_ids != second.procedure_ids
    assert len(procedure_nodes(c)) == 2
    await c.aclose()


async def test_chat_path_idempotency_error_type(tmp_storage, fake_embed):
    c = make(tmp_storage, fake_embed, FakeJev())
    await c.add(content="Likes tea", user_id="u", request_id="r")
    with pytest.raises(IdempotencyConflictError):
        await c.add(content="Likes coffee", user_id="u", request_id="r")
    await c.aclose()
