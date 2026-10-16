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
from dual_mem.storage.cache_store import CacheStore
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


async def test_crash_after_commit_keeps_relation_review(
    tmp_storage, fake_embed, monkeypatch
):
    c = make(tmp_storage, fake_embed, FakeJev(need_context_script))
    seed_head(c, fake_embed, memory_at=None)

    def crash(self, *args, **kwargs):
        raise ProcessDied("process died after commit")

    monkeypatch.setattr(MemAgent, "_committed", crash)
    with pytest.raises(ProcessDied):
        await c.add(content="Lives in Beijing now", user_id="u", request_id="r1")
    monkeypatch.undo()
    # The version committed, and its review intent was queued before the commit.
    (provisional,) = [
        n
        for n in c.factory.vector.get_many({"layer": Layer.L1_FACT.value}, limit=10)
        if n.custom.get("needs_review")
    ]
    (queued,) = c.pending_reviews()
    assert queued["node_id"] == provisional.node_id
    await c.aclose()


async def test_crash_after_commit_with_lost_queue_recovers_on_retry(
    tmp_storage, fake_embed, monkeypatch
):
    """Even if the queue row is gone, the retried request restores it from the plan."""
    jev = FakeJev(need_context_script)
    c = make(tmp_storage, fake_embed, jev)
    seed_head(c, fake_embed, memory_at=None)

    def crash(self, *args, **kwargs):
        raise ProcessDied("process died after commit")

    monkeypatch.setattr(MemAgent, "_committed", crash)
    with pytest.raises(ProcessDied):
        await c.add(content="Lives in Beijing now", user_id="u", request_id="r1")
    monkeypatch.undo()
    c.factory.cache.conn.execute("DELETE FROM review_queue")
    calls = len(jev.calls)

    retried = await c.add(content="Lives in Beijing now", user_id="u", request_id="r1")
    assert len(jev.calls) == calls  # replayed from the plan journal, not re-decided
    assert retried.status == "pending_review"
    assert retried.decisions[0].review_reason == "NEED_CONTEXT"
    (queued,) = c.pending_reviews()
    assert [queued["node_id"]] == retried.needs_review == retried.extra_node_ids

    jev.responses = {"p0:target": "head", "p0:action": "ADD_OVERRIDE"}
    assert (await c.digest()).reviewed == 1
    assert not c.factory.vector.get("head").is_latest
    await c.aclose()


async def test_review_of_uncommitted_intent_waits_then_resolves(
    tmp_storage, fake_embed
):
    """An intent queued for a write that has not committed is retried, not dropped."""
    c = make(tmp_storage, fake_embed, FakeJev(), review_max_attempts=3)
    c.factory.cache.enqueue_review(
        kind="relation",
        app_id="default",
        user_id="u",
        agent_id="",
        node_id="never-committed",
        payload={"reason": "NEED_CONTEXT", "request_key": "write:raw:req"},
    )
    digest = await c.digest()
    assert digest.success and digest.reviewed == 0
    (pending,) = c.pending_reviews()
    assert pending["attempts"] == 1 and "not committed" in pending["error"]
    await c.aclose()


async def test_review_of_replanned_write_resolves_without_change(
    tmp_storage, fake_embed
):
    """The write committed without this provisional node: the intent is moot."""
    c = make(tmp_storage, fake_embed, FakeJev())
    written = await c.add(content="Likes tea", user_id="u", request_id="r1")
    c.factory.cache.enqueue_review(
        kind="relation",
        app_id="default",
        user_id="u",
        agent_id="",
        node_id="dropped-by-replan",
        payload={
            "reason": "NEED_CONTEXT",
            "request_key": f"write:{written.memory_id}:r1",
        },
    )
    assert (await c.digest()).reviewed == 1 and c.pending_reviews() == []
    await c.aclose()


def test_requeue_resets_attempts(tmp_path):
    cache = CacheStore(str(tmp_path))
    args = dict(kind="relation", app_id="a", user_id="u", agent_id="", node_id="n")
    cache.enqueue_review(**args, payload={})
    (item,) = cache.list_reviews()
    cache.fail_review(item["id"], error="in flight", max_attempts=1)
    assert cache.list_reviews() == []
    cache.enqueue_review(**args, payload={"again": True})
    (item,) = cache.list_reviews()
    assert item["attempts"] == 0 and item["error"] == ""


# --- 5. crash after review re-extraction commits ---------------------------------


async def test_review_extract_replays_after_crash_before_resolve(
    tmp_storage, fake_embed, monkeypatch
):
    jev = FakeJev({"value": "REEXTRACT"})
    c = make(tmp_storage, fake_embed, jev, extract={"facts": []})
    written = await c.add(content="Moved to Beijing", user_id="u")
    c.factory.llm.responses["extract"] = {"facts": [{"content": "Moved to Beijing"}]}
    jev.responses = None

    def crash(self, review_id):
        raise ProcessDied("died before resolve_review")

    monkeypatch.setattr(CacheStore, "resolve_review", crash)
    with pytest.raises(ProcessDied):
        await c.digest()
    monkeypatch.undo()
    committed = c.factory.vector.get_many({"layer": Layer.L1_FACT.value}, limit=10)
    assert len(committed) == 1

    # A different verdict on retry must not create a second version or fail.
    jev.responses = {"p0:action": "RAW_ONLY"}
    calls = len(jev.calls)
    digest = await c.digest()
    assert digest.reviewed == 1 and c.pending_reviews() == []
    assert c.pending_reviews(status="quarantined") == []
    assert len(jev.calls) == calls
    after = c.factory.vector.get_many({"layer": Layer.L1_FACT.value}, limit=10)
    assert [n.node_id for n in after] == [n.node_id for n in committed]
    assert after[0].custom["source_ids"] == [written.memory_id]
    await c.aclose()


# --- 3. tool-message path idempotency --------------------------------------------


def tool_messages(solution="Use BEGIN IMMEDIATE to validate before commit"):
    return [
        {"role": "user", "content": "Fix the SDK transaction"},
        {"role": "tool", "content": solution, "tool_call_id": "tx"},
    ]


def coding_client(tmp_storage, embed, jev=None):
    return MemoryClient(
        settings=Settings(storage_dir=tmp_storage, coding_enabled=True),
        embed=embed,
        llm=FakeLLMClient(
            {
                "json": [
                    {
                        "task": "How to publish transactions",
                        "solution": "Use BEGIN IMMEDIATE to validate before commit",
                        "search_keys": [],
                    }
                ]
            }
        ),
        jev=jev
        or FakeJev(
            {
                "scene": "CODING",
                "coverage": "VALID",
                "0:action": "ADD_SUPPLEMENT",
                "0:target": "NONE",
            }
        ),
    )


def coding_rows(c):
    store = c._coding_writer.store
    with store._lock:
        return store._conn.execute(
            "SELECT memory_id, published FROM coding_memory_meta"
        ).fetchall()


async def test_tool_path_request_id_replays_without_new_memories(
    tmp_storage, fake_embed
):
    c = coding_client(tmp_storage, fake_embed)
    first = await c.add(messages=tool_messages(), user_id="u", request_id="t1")
    calls = len(c.factory.jev.calls), len(c.factory.llm.calls)
    again = await c.add(messages=tool_messages(), user_id="u", request_id="t1")
    assert (len(c.factory.jev.calls), len(c.factory.llm.calls)) == calls
    assert again.memory_id == first.memory_id
    assert again.extra_node_ids == first.extra_node_ids
    assert len(coding_rows(c)) == 1
    raws = [
        n for n in c.factory.vector.get_many({"layer": "L0_RAW"}, limit=10)
        if "coding" in n.tags
    ]
    assert len(raws) == 1
    await c.aclose()


async def test_tool_path_request_id_reuse_with_other_input_is_rejected(
    tmp_storage, fake_embed
):
    c = coding_client(tmp_storage, fake_embed)
    await c.add(messages=tool_messages(), user_id="u", request_id="t1")
    with pytest.raises(IdempotencyConflictError, match="different input"):
        await c.add(
            messages=tool_messages("Use WAL instead"), user_id="u", request_id="t1"
        )
    assert len(coding_rows(c)) == 1
    await c.aclose()


def test_tool_path_request_id_conflict_is_409_over_rest(tmp_storage, fake_embed):
    from fastapi.testclient import TestClient

    from dual_mem.api import create_app

    c = coding_client(tmp_storage, fake_embed)
    api = TestClient(create_app(client=c, settings=c.settings))
    body = {"messages": tool_messages(), "user_id": "u", "request_id": "t1"}
    assert api.post("/v1/memories/", json=body).status_code == 200
    replay = api.post("/v1/memories/", json=body)
    assert replay.status_code == 200
    conflict = api.post(
        "/v1/memories/", json=body | {"messages": tool_messages("Use WAL instead")}
    )
    assert conflict.status_code == 409
    assert conflict.json()["error_type"] == "IdempotencyConflictError"


# --- coverage review vs. the degraded write it belongs to -----------------------


def coverage_rejecting_jev():
    """Coverage always INVALID, so every add degrades and queues a source review."""
    return FakeJev({"coverage": "INVALID"})


def facts(c):
    return c.factory.vector.get_many({"layer": Layer.L1_FACT.value}, limit=50)


async def test_coverage_review_waits_for_its_write_to_commit(
    tmp_storage, fake_embed, monkeypatch
):
    """digest() during the degraded commit must not write a second set."""
    import asyncio

    c = make(
        tmp_storage,
        fake_embed,
        coverage_rejecting_jev(),
        extract={"facts": [{"content": "User is named Alice"}]},
        extract_coverage_retries=0,
    )
    gate, entered = asyncio.Event(), asyncio.Event()

    async def held_to_thread(fn, *args, **kwargs):
        # Bound methods are new objects on every access: compare by name.
        if getattr(fn, "__name__", "") == "apply_version_plan" and kwargs.get("nodes"):
            entered.set()
            await gate.wait()
        return fn(*args, **kwargs)

    monkeypatch.setattr("dual_mem.agent.mem_agent.asyncio.to_thread", held_to_thread)
    write = asyncio.create_task(
        c.add(content="My name is Alice and I moved to Shanghai", user_id="u")
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    # The degraded write's review is queued, its plan is not committed yet.
    (queued,) = c.pending_reviews()
    assert queued["kind"] == "extract"
    during = await c.digest()
    assert during.reviewed == 0 and facts(c) == []
    (waiting,) = c.pending_reviews()
    assert "has not completed" in waiting["error"]

    gate.set()
    result = await asyncio.wait_for(write, timeout=5)
    assert result.status == "pending_review"
    (fact,) = facts(c)
    # Committing reopened the item; only now does the review run.
    (reopened,) = c.pending_reviews()
    assert reopened["attempts"] == 0
    monkeypatch.undo()
    # The review sees the committed fact and judges its re-extraction already known.
    c.factory.jev.responses = {"p0:target": fact.node_id, "p0:action": "SKIP"}
    after = await c.digest()
    assert after.reviewed == 1 and c.pending_reviews() == []
    review_call = c.factory.jev.calls[-1]
    assert [h["id"] for h in review_call["state"]["current"]] == [fact.node_id]
    assert [f.node_id for f in facts(c)] == [fact.node_id]
    await c.aclose()


async def test_coverage_review_after_stale_failure_waits_for_the_retry(
    tmp_storage, fake_embed, monkeypatch
):
    """Degraded commit fails concurrently; digest must not write before the retry."""
    from dual_mem import StaleDecisionError

    c = make(
        tmp_storage,
        fake_embed,
        coverage_rejecting_jev(),
        extract={"facts": [{"content": "User is named Alice"}]},
        extract_coverage_retries=0,
    )
    store = type(c.factory.vector)
    original = store.apply_version_plan
    monkeypatch.setattr(store, "apply_version_plan", lambda self, **kw: False)
    with pytest.raises(StaleDecisionError):
        await c.add(content="My name is Alice", user_id="u", request_id="r1")
    monkeypatch.setattr(store, "apply_version_plan", original)
    assert len(c.pending_reviews()) == 1

    calls = len(c.factory.llm.calls)
    digest = await c.digest()
    assert digest.reviewed == 0 and facts(c) == []
    assert len(c.factory.llm.calls) == calls  # no re-extraction while unresolved

    retried = await c.add(content="My name is Alice", user_id="u", request_id="r1")
    assert retried.status == "pending_review" and len(facts(c)) == 1
    (item,) = c.pending_reviews()
    assert item["attempts"] == 0
    await c.aclose()


async def test_coverage_review_is_moot_when_retry_covered_the_source(
    tmp_storage, fake_embed, monkeypatch
):
    from dual_mem import StaleDecisionError

    jev = coverage_rejecting_jev()
    c = make(
        tmp_storage,
        fake_embed,
        jev,
        extract={"facts": [{"content": "User is named Alice"}]},
        extract_coverage_retries=0,
    )
    store = type(c.factory.vector)
    original = store.apply_version_plan
    monkeypatch.setattr(store, "apply_version_plan", lambda self, **kw: False)
    with pytest.raises(StaleDecisionError):
        await c.add(content="My name is Alice", user_id="u", request_id="r1")
    monkeypatch.setattr(store, "apply_version_plan", original)

    jev.responses = None  # the retry's coverage is accepted
    retried = await c.add(content="My name is Alice", user_id="u", request_id="r1")
    assert retried.status == "committed" and len(facts(c)) == 1
    calls = len(c.factory.llm.calls)
    digest = await c.digest()
    assert digest.reviewed == 1 and c.pending_reviews() == []
    assert len(c.factory.llm.calls) == calls and len(facts(c)) == 1
    await c.aclose()


async def test_reextract_request_replays_and_review_runs_after_it(
    tmp_storage, fake_embed
):
    """A missed extraction completes its request too; retries replay it."""
    jev = FakeJev({"value": "REEXTRACT"})
    c = make(tmp_storage, fake_embed, jev, extract={"facts": []})
    first = await c.add(content="Moved to Beijing", user_id="u", request_id="r1")
    c.factory.cache.conn.execute("DELETE FROM request_outcomes")
    calls = len(jev.calls), len(c.factory.llm.calls)
    again = await c.add(content="Moved to Beijing", user_id="u", request_id="r1")
    assert (len(jev.calls), len(c.factory.llm.calls)) == calls
    assert again.status == "pending_review" and again.needs_review == first.needs_review
    assert again.decisions[0].review_reason == "REEXTRACT" and not again.is_ephemeral
    await c.aclose()


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
    coding_store = c._coding_writer.store
    original = coding_store._publish

    def die(*args, **kwargs):
        raise ProcessDied("died before publish")

    monkeypatch.setattr(coding_store, "_publish", die)
    with pytest.raises(ProcessDied):
        await c.add(messages=tool_messages(), user_id="u", request_id="t1")
    assert [row["published"] for row in coding_rows(c)] == [0]
    monkeypatch.setattr(coding_store, "_publish", original)

    result = await c.add(messages=tool_messages(), user_id="u", request_id="t1")
    rows = coding_rows(c)
    assert [row["published"] for row in rows] == [1]
    assert result.extracted_count == 1
    assert coding_store.request_result(next(iter(_request_keys(coding_store))))
    await c.aclose()


def _request_keys(store):
    with store._lock:
        return [r[0] for r in store._conn.execute("SELECT request_key FROM coding_requests")]


async def test_tool_path_without_request_id_records_nothing(tmp_storage, fake_embed):
    """A generated request_id can never be replayed, so nothing is recorded for it."""
    c = coding_client(tmp_storage, fake_embed)
    first = await c.add(messages=tool_messages(), user_id="u")
    second = await c.add(messages=tool_messages(), user_id="u")
    assert first.memory_id != second.memory_id
    assert _request_keys(c._coding_writer.store) == []
    await c.aclose()


async def test_coding_request_records_expire_on_digest(tmp_storage, fake_embed):
    c = coding_client(tmp_storage, fake_embed)
    await c.add(messages=tool_messages(), user_id="u", request_id="old")
    coding_store = c._coding_writer.store
    with coding_store._lock, coding_store._conn:
        coding_store._conn.execute("UPDATE coding_requests SET ts = 0")
    await c.add(
        messages=tool_messages("Use WAL instead"), user_id="u", request_id="fresh"
    )
    await c.digest()
    keys = _request_keys(coding_store)
    assert len(keys) == 1 and keys[0].endswith('"fresh"]')
    await c.aclose()


async def test_chat_path_idempotency_error_type(tmp_storage, fake_embed):
    c = make(tmp_storage, fake_embed, FakeJev())
    await c.add(content="Likes tea", user_id="u", request_id="r")
    with pytest.raises(IdempotencyConflictError):
        await c.add(content="Likes coffee", user_id="u", request_id="r")
    await c.aclose()
