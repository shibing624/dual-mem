"""Fully mocked coding JEV contracts, publication failures and immutable history."""

import asyncio
import copy
import json
from dataclasses import replace

import pytest
from conftest import FakeEmbedService, FakeJev, FakeLLMClient

from dual_mem.coding import (
    CodingConcurrentWriteError,
    CodingDecisionError,
    CodingMemory,
    CodingMemoryDraft,
    CodingMemoryExtractor,
    CodingMemoryReconciler,
    CodingMemoryStore,
    CodingWriter,
)
from dual_mem.types import Layer, MemoryNode


class FakeVector:
    """In-memory VDB with immutable upsert and injectable partial acknowledgements."""

    def __init__(self):
        self.nodes = {}
        self.fail_upsert = None
        self.fail_patch = None
        self.query_calls = []

    def get(self, node_id):
        return copy.deepcopy(self.nodes.get(node_id))

    def get_by_ids(self, node_ids):
        return {nid: self.get(nid) for nid in node_ids if nid in self.nodes}

    def upsert(self, nodes):
        for i, node in enumerate(nodes):
            if self.fail_upsert == "silent" and i:
                return
            if self.fail_upsert == "partial" and i:
                raise OSError("partial vector append")
            old = self.nodes.get(node.node_id)
            if old:
                if old.content != node.content:
                    raise ValueError("immutable vector content")
                continue
            self.nodes[node.node_id] = copy.deepcopy(node)
        if self.fail_upsert == "after":
            raise OSError("vector append acknowledgement lost")

    def update_payload_many(self, patches):
        for i, (nid, patch) in enumerate(patches.items()):
            if self.fail_patch == "silent":
                return
            meta = self.nodes[nid].to_metadata()
            meta.update(patch)
            node = self.nodes[nid]
            self.nodes[nid] = MemoryNode.from_storage(
                node.content, meta, node.embedding
            )
            if self.fail_patch == "partial" and i == 0:
                raise OSError("partial vector links")

    def query(self, *, embedding, where, top_k):
        self.query_calls.append({"where": where, "top_k": top_k})
        out = []
        for node in self.nodes.values():
            meta = node.to_metadata()
            if all(
                meta.get(k) in v["$in"] if isinstance(v, dict) else meta.get(k) == v
                for k, v in where.items()
            ):
                hit = copy.deepcopy(node)
                hit.score = 1.0
                out.append(hit)
        return out[:top_k]


@pytest.fixture
def store(tmp_path, fake_embed):
    result = CodingMemoryStore(
        db_path=str(tmp_path / "coding.db"), vector=FakeVector(), embed=fake_embed
    )
    yield result
    result.close()


def memory(mid="old", **kw):
    return CodingMemory(
        **{
            "memory_id": mid,
            "user_id": "u",
            "task": "How to build?",
            "solution": "Use compiler A; keep flag --safe",
            **kw,
        }
    )


def candidate(**kw):
    return CodingMemoryDraft(
        **{
            "user_id": "u",
            "task": "How to build?",
            "solution": "Use compiler B; keep flag --safe",
            **kw,
        }
    )


def source(text="Use compiler B; keep flag --safe"):
    return [
        {"role": "user", "content": text},
        {"role": "tool", "content": text, "tool_call_id": "call-1"},
    ]


def writer(store, *, candidates=None, answers=None, llm=None):
    jev = FakeJev(
        {
            "scene": "CODING",
            **{
                f"{i}:action": "ADD_SUPPLEMENT"
                for i in range(len(candidates) if isinstance(candidates, list) else 1)
            },
            **(answers or {}),
        }
    )
    llm = llm or FakeLLMClient(
        {"json": candidates if candidates is not None else [candidate().to_dict()]}
    )
    return CodingWriter(
        store=store,
        extractor=CodingMemoryExtractor(llm=llm),
        reconciler=CodingMemoryReconciler(jev=jev),
        llm=llm,
        jev=jev,
    )


async def test_write_preserves_full_source_and_batches_all_decisions(store):
    messages = source("前" * 800 + "\n--critical-middle=42\n" + "尾" * 2400)
    messages.insert(
        1,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"file_path": "/src/main.py"}),
                    },
                }
            ],
        },
    )
    proposals = [candidate().to_dict(), candidate().to_dict()]
    w = writer(store, candidates=proposals)
    result = await w.write(messages, user_id="u")
    assert result["success"] and result["commit_passed"]
    assert len(result["memory_ids"]) == 2
    assert result["extra_node_ids"] == [f"{mid}_key0" for mid in result["memory_ids"]]
    assert len(w.jev.calls) == 2 and len(w.llm.calls) == 1
    assert set(w.jev.calls[-1]["questions"]) == {
        "coverage",
        "0:action",
        "1:action",
    }
    assert w.jev.calls[-1]["state"]["source"] == messages
    assert json.loads(w.llm.calls[0]["user"]) == messages
    raw = store.get_source(result["raw_id"], user_id="u")
    assert raw["messages"] == messages and raw["indexed"] == 1
    assert store.vector.get(result["raw_id"]).layer == Layer.L0_RAW
    for mid in result["memory_ids"]:
        assert store.get(mid).source_ids == [result["raw_id"]]
        assert store.get(mid).files == ["/src/main.py"]
    assert store.get_source(result["raw_id"], user_id="other") is None


@pytest.mark.parametrize("coverage,raises", [("RAW_ONLY", False), ("MISSED", True)])
async def test_empty_extraction_requires_jev_coverage_and_retains_l0(
    store, coverage, raises
):
    w = writer(store, candidates=[], answers={"coverage": coverage})
    if raises:
        with pytest.raises(CodingDecisionError, match="coverage"):
            await w.write(source(), user_id="u")
    else:
        result = await w.write(source(), user_id="u")
        assert result["success"] and not result["commit_passed"]
        assert (
            result["memory_ids"] == []
            and result["decisions"][0]["action"] == "RAW_ONLY"
        )
    assert len(w.jev.calls) == 2
    assert store.list_by_user(user_id="u") == []
    assert store._conn.execute("SELECT count(*) FROM coding_sources").fetchone()[0] == 1


async def test_return_to_ancestor_uses_current_target_with_full_history(store):
    await store.add(memory("a"))
    await store.add(
        memory("b", solution="Use compiler B; keep flag --safe", supersedes=["a"])
    )
    w = writer(
        store,
        candidates=[candidate(solution="Use compiler A; keep flag --safe").to_dict()],
        answers={"0:action": "ADD_OVERRIDE", "0:target": "b"},
    )
    result = await w.write(
        source("Use compiler A again; keep flag --safe"), user_id="u"
    )
    state = w.jev.calls[-1]["state"]
    assert [m["memory_id"] for m in state["current"]] == ["b"]
    assert state["current"][0]["history_versions"][0]["solution"] == memory().solution
    target_call = next(
        call for call in w.jev.calls if "0:target" in call["questions"]
    )
    assert "a" not in target_call["questions"]["0:target"]["criteria"]
    assert "ADD_SUPPLEMENT" not in w.jev.calls[-1]["questions"]["0:action"]["criteria"]
    new_id = result["memory_ids"][0]
    assert store.get(new_id).supersedes == ["b"]
    assert store.get("a").solution == memory().solution
    assert store.get("b").superseded_by == [new_id]
    assert not store.get("a").is_latest and not store.get("b").is_latest
    assert store.vector.get("b_key0").superseded_by == [f"{new_id}_key0"]
    assert store.vector.get(f"{new_id}_key0").supersedes == ["b_key0"]
    assert len(store.snapshot(user_id="u")[2][new_id]) == 2


async def test_delete_archives_current_without_new_row(store):
    await store.add(memory())
    w = writer(store, answers={"0:action": "DELETE", "0:target": "old"})
    result = await w.write(source(), user_id="u")
    assert result["memory_ids"] == []
    assert result["decisions"][0]["action"] == "DELETE"
    assert result["decisions"][0]["targets"] == ["old"]
    assert store.get("old").is_latest is False
    assert store.get("old").superseded_by == []
    assert store.list_by_user(user_id="u") == []


async def test_none_target_rejects_scripted_skip(store):
    await store.add(memory())
    w = writer(store, answers={"0:action": "SKIP", "0:target": "NONE"})
    with pytest.raises(CodingDecisionError):
        await w.write(source(), user_id="u")
    action_call = next(call for call in w.jev.calls if "0:action" in call["questions"])
    assert set(action_call["questions"]["0:action"]["criteria"]) == {
        "ADD_SUPPLEMENT",
        "ADD_TEMPORAL",
        "RAW_ONLY",
        "INVALID",
        "NEED_CONTEXT",
    }
    assert store.get("old").is_latest


async def test_selected_target_skip_has_no_repair_question(store):
    await store.add(memory())
    w = writer(store, answers={"0:action": "SKIP", "0:target": "old"})
    result = await w.write(source(), user_id="u")
    assert result["memory_ids"] == []
    assert result["decisions"][0]["action"] == "SKIP"
    assert result["decisions"][0]["targets"] == ["old"]
    criteria = next(
        call["questions"]["0:action"]["criteria"]
        for call in w.jev.calls
        if "0:action" in call["questions"]
    )
    assert "SKIP" in criteria and "ADD_SUPPLEMENT" not in criteria
    assert all(
        not any(str(name).endswith(":repair") for name in call["questions"])
        for call in w.jev.calls
    )


@pytest.mark.parametrize("action", ["SKIP", "DELETE", "ADD_CONFLICT"])
async def test_relation_cannot_cross_version_boundary(store, action):
    await store.add(memory(workspace_id="old-workspace"))
    w = writer(
        store,
        candidates=[candidate(workspace_id="new-workspace").to_dict()],
        answers={"0:action": action, "0:target": "old"},
    )
    with pytest.raises(CodingDecisionError, match="boundary"):
        await w.write(source(), user_id="u")
    assert store.get("old").is_latest
    assert len(store.list_by_user(user_id="u")) == 1


async def test_multiple_target_unsure_does_not_retire_current(store):
    await store.add(memory())

    def decide(*, state, questions):
        if any(str(name).endswith(":old") for name in questions):
            return {name: "UNSURE" for name in questions}
        return {
            "scene": "CODING",
            "coverage": "VALID",
            "0:target": "MULTIPLE",
            "0:action": "ADD_OVERRIDE",
        }

    w = writer(store)
    w.jev.responses = decide
    with pytest.raises(CodingDecisionError, match="uncertain"):
        await w.write(source(), user_id="u")
    assert store.get("old").is_latest


async def test_multiple_target_must_resolve_to_one_head(store):
    await store.add_many([memory("a"), memory("b")])

    def decide(*, state, questions):
        if any("INCLUDE" in question["criteria"] for question in questions.values()):
            return {name: "INCLUDE" for name in questions}
        return {
            "scene": "CODING",
            "coverage": "VALID",
            "0:target": "MULTIPLE",
        }

    w = writer(store)
    w.jev.responses = decide
    with pytest.raises(CodingDecisionError, match="exactly one"):
        await w.write(source(), user_id="u")
    assert store.get("a").is_latest and store.get("b").is_latest


def test_candidate_limit_leaves_room_for_none_and_multiple():
    jev = FakeJev()
    CodingMemoryReconciler(jev=jev, candidate_limit=253)
    with pytest.raises(ValueError, match=r"\[1,253\]"):
        CodingMemoryReconciler(jev=jev, candidate_limit=254)


@pytest.mark.parametrize(
    "action,target",
    [
        ("INVALID", "NONE"),
        ("NEED_CONTEXT", "NONE"),
        ("ADD_OVERRIDE", "NONE"),
        ("ADD_SUPPLEMENT", "old"),
        ("RAW_ONLY", "old"),
        ("ADD_OVERRIDE", "unknown"),
    ],
)
async def test_invalid_decision_fails_without_retiring_current(store, action, target):
    await store.add(memory())
    w = writer(store, answers={"0:action": action, "0:target": target})
    with pytest.raises(CodingDecisionError):
        await w.write(source(), user_id="u")
    assert store.get("old").is_latest
    assert len(store.list_by_user(user_id="u")) == 1


@pytest.mark.parametrize("action", ["SKIP", "RAW_ONLY"])
async def test_known_and_no_value_are_distinct_explicit_decisions(store, action):
    await store.add(memory())
    w = writer(
        store,
        answers={"0:action": action, "0:target": "old" if action == "SKIP" else "NONE"},
    )
    result = await w.write(source(), user_id="u")
    assert result["memory_ids"] == [] and result["decisions"][0]["action"] == action
    assert result["commit_passed"] == (action == "SKIP")
    assert store.get("old").is_latest


@pytest.mark.parametrize(
    "data",
    [
        {},
        "bad",
        [None],
        [{"task": ""}],
        [{"task": "q", "solution": "s", "confidence": float("nan")}],
        [{"task": "q", "solution": "s", "search_keys": "keys"}],
        [{"task": "q", "solution": "s", "boundary_scope": "invalid"}],
    ],
)
async def test_malformed_extraction_propagates_with_retained_source(store, data):
    w = writer(store, candidates=data)
    with pytest.raises(CodingDecisionError):
        await w.write(source(), user_id="u")
    assert store.list_by_user(user_id="u") == []
    assert store._conn.execute("SELECT count(*) FROM coding_sources").fetchone()[0] == 1


async def test_generation_failure_is_not_empty_success(store):
    def fail(**kw):
        raise OSError("generation unavailable")

    w = writer(store, llm=FakeLLMClient({"json": fail}))
    with pytest.raises(OSError, match="generation"):
        await w.write(source(), user_id="u")
    assert len(w.jev.calls) == 1
    assert store._conn.execute("SELECT count(*) FROM coding_sources").fetchone()[0] == 1


async def test_scene_failure_retains_evidence_and_propagates(store):
    w = writer(store)
    w.jev.responses = OSError("scene unavailable")
    with pytest.raises(OSError, match="scene"):
        await w.write(source(), user_id="u")
    assert store._conn.execute("SELECT count(*) FROM coding_sources").fetchone()[0] == 1
    assert w.llm.calls == []


async def test_raw_vector_failure_keeps_sql_source(store):
    store.vector.fail_upsert = "after"
    with pytest.raises(OSError):
        await writer(store).write(source(), user_id="u")
    row = store._conn.execute("SELECT * FROM coding_sources").fetchone()
    assert json.loads(row["messages"]) == source() and row["indexed"] == 0


@pytest.mark.parametrize(
    "failure,error",
    [("partial", OSError), ("after", OSError), ("silent", CodingDecisionError)],
)
async def test_partial_candidate_vectors_never_publish_or_retire(store, failure, error):
    await store.add(memory(search_keys=["compiler flags"]))
    store.vector.fail_upsert = failure
    with pytest.raises(error):
        await store.add(memory("new", search_keys=["build flags"], supersedes=["old"]))
    assert store.get("new") is None and store.get("old").is_latest
    assert [m.memory_id for m in store.list_by_user(user_id="u")] == ["old"]
    assert [m["memory_id"] for m in await store.search(query="build", user_id="u")] == [
        "old"
    ]
    assert store.snapshot(user_id="u")[0] == 1


@pytest.mark.parametrize(
    "failure,error", [("partial", OSError), ("silent", CodingDecisionError)]
)
async def test_link_failure_sql_remains_authority_and_later_append_repairs_projection(
    store, failure, error
):
    await store.add(memory(search_keys=["compiler flags"]))
    store.vector.fail_patch = failure
    with pytest.raises(error):
        await store.add(memory("failed", supersedes=["old"]))
    assert store.get("failed") is None and store.get("old").is_latest
    assert store.get("old").superseded_by == []
    hits = await store.search(query="build", user_id="u", top_k=1)
    assert [m["memory_id"] for m in hits] == ["old"]
    assert "is_latest" not in store.vector.query_calls[-1]["where"]
    store.vector.fail_patch = None
    await store.add(memory("next", supersedes=["old"]))
    assert store.get("old").superseded_by == ["next"]
    assert store.vector.get("old_key0").superseded_by == ["next_key0"]


@pytest.mark.parametrize("bad", [[], [[1.0]], [[float("inf")], [1.0]]])
async def test_embedding_shape_or_values_fail_before_publication(store, bad):
    await store.add(memory())

    class BadEmbed:
        async def embed_batch(self, texts):
            return bad

    store.embed = BadEmbed()
    with pytest.raises(CodingDecisionError):
        await store.add(memory("bad", search_keys=["q"], supersedes=["old"]))
    assert store.get("bad") is None and store.get("old").is_latest


async def test_duplicate_target_batch_rejected_before_any_publication(store):
    await store.add(memory())
    w = writer(
        store,
        candidates=[candidate().to_dict(), candidate().to_dict()],
        answers={
            "0:action": "ADD_OVERRIDE",
            "0:target": "old",
            "1:action": "ADD_OVERRIDE",
            "1:target": "old",
        },
    )
    with pytest.raises(CodingConcurrentWriteError):
        await w.write(source(), user_id="u")
    assert len(store.list_by_user(user_id="u")) == 1 and store.get("old").is_latest
    assert (
        store._conn.execute(
            "SELECT count(*) FROM coding_memory_meta WHERE published=0"
        ).fetchone()[0]
        == 0
    )


async def test_multi_draft_batch_is_atomic_on_vector_failure(store):
    store.vector.fail_upsert = "partial"
    with pytest.raises(OSError):
        await store.add_many([memory("one"), memory("two")])
    assert store.get("one") is None and store.get("two") is None
    assert store.snapshot(user_id="u")[0] == 0


async def test_stale_empty_scope_epoch_cannot_publish_parallel_add(store):
    second = CodingMemoryStore(
        db_path=store.db_path, vector=store.vector, embed=store.embed
    )
    try:
        epoch = store.snapshot(user_id="u")[0]
        await store.add_many([memory("winner")], expected_epoch=epoch)
        with pytest.raises(CodingConcurrentWriteError):
            await second.add_many([memory("loser")], expected_epoch=epoch)
        assert second.get("loser") is None
    finally:
        second.close()


async def test_cas_rechecks_after_vector_preparation(store):
    second = CodingMemoryStore(
        db_path=store.db_path, vector=store.vector, embed=FakeEmbedService()
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class DelayedEmbed(FakeEmbedService):
        async def embed_batch(self, texts):
            entered.set()
            await release.wait()
            return await super().embed_batch(texts)

    store.embed = DelayedEmbed()
    task = asyncio.create_task(store.add_many([memory("loser")], expected_epoch=0))
    try:
        await entered.wait()
        await second.add(memory("winner"))
        release.set()
        with pytest.raises(CodingConcurrentWriteError):
            await task
        assert store.get("loser") is None
        assert [m.memory_id for m in store.list_by_user(user_id="u")] == ["winner"]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        second.close()


@pytest.mark.parametrize(
    "change",
    [
        {"user_id": "other"},
        {"app_id": "other"},
        {"agent_id": "other"},
        {"workspace_id": "other"},
        {"branch": "other"},
        {"boundary_scope": "strict"},
    ],
)
async def test_retirement_cannot_cross_scope_or_structural_boundary(store, change):
    await store.add(memory())
    new = replace(memory("new", supersedes=["old"]), **change)
    with pytest.raises(ValueError):
        await store.add(new)
    assert store.get("old").is_latest and store.get("new") is None


async def test_strict_session_retirement_rejected_but_project_session_supported(store):
    await store.add(memory("strict", boundary_scope="strict", session_id="a"))
    with pytest.raises(ValueError, match="boundary"):
        await store.add(
            memory(
                "strict-new",
                boundary_scope="strict",
                session_id="b",
                supersedes=["strict"],
            )
        )
    await store.add(memory("project", session_id="a"))
    await store.add(memory("project-new", session_id="b", supersedes=["project"]))
    assert not store.get("project").is_latest


async def test_immutable_ids_update_delete_and_retired_targets_rejected(store):
    await store.add(memory())
    for operation in (
        store.add(memory(solution="changed")),
        store.update(memory()),
        store.delete("old"),
    ):
        with pytest.raises(ValueError):
            await operation
    await store.add(memory("new", supersedes=["old"]))
    with pytest.raises(CodingConcurrentWriteError):
        await store.add(memory("other", supersedes=["old"]))
    assert store.get("old").solution == memory().solution


async def test_scope_snapshot_has_no_100_row_truncation_and_semantic_selection_is_bounded(
    store,
):
    await store.add_many([memory(f"m{i:03d}") for i in range(105)])
    epoch, heads, histories = store.snapshot(user_id="u")
    assert len(heads) == 105 and epoch == 1

    def decide(state, questions):
        if any(str(name).endswith(":action") for name in questions):
            return {"0:action": "ADD_OVERRIDE"}
        if "coverage" in questions:
            return {"coverage": "VALID", "0:target": "m104"}
        return {qid: "INCLUDE" if qid == "m104" else "EXCLUDE" for qid in questions}

    jev = FakeJev(decide)
    (op,) = await CodingMemoryReconciler(jev=jev, candidate_limit=32).reconcile(
        [candidate()], heads, source_messages=source(), history_versions=histories
    )
    assert op.target_memory_id == "m104" and len(jev.calls) == 6
    store.SCAN_LIMIT = 100
    with pytest.raises(CodingDecisionError, match="scan budget"):
        store.snapshot(user_id="u")


async def test_provider_request_rejection_retains_source(store):
    w = writer(store)
    w.jev.responses = ValueError("context budget exceeded")
    with pytest.raises(CodingDecisionError, match="request rejected"):
        await w.write(source("x" * 1200), user_id="u")
    assert store._conn.execute("SELECT count(*) FROM coding_sources").fetchone()[0] == 1


async def test_evaluate_only_custom_provider_is_supported(store):
    class MinimalProvider:
        def __init__(self, delegate):
            self.delegate = delegate

        async def evaluate(self, *, state, questions):
            return await self.delegate.evaluate(state=state, questions=questions)

    w = writer(store)
    provider = MinimalProvider(w.jev)
    w.jev = provider
    w.reconciler.jev = provider
    result = await w.write(source(), user_id="u")
    assert result["success"] and result["commit_passed"]


async def test_missing_answer_coverage_fails_closed(store):
    class IncompleteJev(FakeJev):
        async def evaluate(self, **kw):
            answers = await super().evaluate(**kw)
            answers.pop(next(iter(answers)))
            return answers

    rec = CodingMemoryReconciler(jev=IncompleteJev())
    with pytest.raises(CodingDecisionError, match="question coverage"):
        await rec.reconcile([candidate()], [], source_messages=source())


@pytest.mark.parametrize("broken", ["missing", "foreign", "cycle"])
async def test_reopen_rejects_corrupted_history(store, broken):
    await store.add_many([memory("a"), memory("b")])
    if broken == "missing":
        store._conn.execute(
            "UPDATE coding_memory_meta SET supersedes='[\"absent\"]' WHERE memory_id='b'"
        )
    else:
        store._conn.execute(
            "UPDATE coding_memory_meta SET supersedes='[\"a\"]' WHERE memory_id='b'"
        )
        if broken == "foreign":
            store._conn.execute(
                "UPDATE coding_memory_meta SET user_id='other' WHERE memory_id='a'"
            )
        else:
            store._conn.execute(
                "UPDATE coding_memory_meta SET supersedes='[\"b\"]' WHERE memory_id='a'"
            )
    store._conn.commit()
    with pytest.raises(CodingDecisionError):
        CodingMemoryStore(db_path=store.db_path, vector=store.vector, embed=store.embed)


async def test_reopen_rejects_one_way_links_without_rewriting(store):
    await store.add_many([memory("a"), memory("b")])
    store._conn.execute(
        "UPDATE coding_memory_meta SET supersedes='[\"a\"]' WHERE memory_id='b'"
    )
    store._conn.commit()
    with pytest.raises(CodingDecisionError, match="inconsistent"):
        CodingMemoryStore(db_path=store.db_path, vector=store.vector, embed=store.embed)
    assert store.get("a").is_latest
    assert store.get("a").superseded_by == []


async def test_live_snapshot_rejects_missing_history_and_reverse_links(store):
    await store.add(memory("a"))
    await store.add(memory("b", supersedes=["a"]))
    store._conn.execute(
        "UPDATE coding_memory_meta SET superseded_by='[]' WHERE memory_id='a'"
    )
    store._conn.commit()
    with pytest.raises(CodingDecisionError, match="reverse"):
        store.snapshot(user_id="u")


async def test_pending_id_has_no_replay_and_remains_invisible_after_reopen(store):
    store.vector.fail_upsert = "after"
    with pytest.raises(OSError):
        await store.add(memory("pending"))
    store.vector.fail_upsert = None
    with pytest.raises(ValueError, match="immutable"):
        await store.add(memory("pending"))
    second = CodingMemoryStore(
        db_path=store.db_path, vector=store.vector, embed=store.embed
    )
    try:
        assert second.get("pending") is None
        assert await second.search(query="build", user_id="u") == []
    finally:
        second.close()


async def test_search_deduplicates_keys_and_preserves_scope(store):
    await store.add(memory("a", search_keys=["one", "two"]))
    await store.add(replace(memory("foreign"), app_id="other"))
    await store.add(memory("b"))
    hits = await store.search(query="build", user_id="u", top_k=2)
    assert [h["memory_id"] for h in hits] == ["a", "b"]
    assert all(h["app_id"] == "default" for h in hits)
    assert await store.search(query="build", user_id="other") == []


@pytest.mark.parametrize("histories", [None, {"b": []}])
async def test_direct_reconcile_cannot_skip_required_history(store, histories):
    await store.add(memory("a"))
    await store.add(memory("b", supersedes=["a"]))
    with pytest.raises(CodingDecisionError, match="history_versions"):
        await CodingMemoryReconciler(jev=FakeJev()).reconcile(
            [candidate()],
            [store.get("b")],
            source_messages=source(),
            history_versions=histories,
        )
