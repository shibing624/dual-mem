from dual_mem.storage.cache_store import CacheStore
from dual_mem.storage.history_store import HistoryStore


def test_profile_roundtrip(tmp_storage):
    c = CacheStore(tmp_storage)
    assert c.get_profile("u::a::s") is None
    c.set_profile("u::a::s", {"name": "Alice"})
    assert c.get_profile("u::a::s") == {"name": "Alice"}


def test_s2_queue_enqueue_dequeue(tmp_storage):
    c = CacheStore(tmp_storage)
    assert c.dequeue_s2_task() is None
    c.enqueue_s2_task("u1", "app1")
    c.enqueue_s2_task("u2", "app1")
    t1 = c.dequeue_s2_task()
    assert t1["user_id"] == "u1"
    t2 = c.dequeue_s2_task()
    assert t2["user_id"] == "u2"
    assert c.dequeue_s2_task() is None


def test_pipeline_log(tmp_storage):
    c = CacheStore(tmp_storage)
    c.log_pipeline(request_id="r1", stage="EXTRACT", payload={"facts": ["a"]})
    c.log_pipeline(request_id="r1", stage="RECONCILE", payload={"ops": []})
    c.log_pipeline(request_id="r2", stage="EXTRACT", payload={"facts": []})

    logs = c.list_pipeline_logs("r1")
    assert len(logs) == 2
    assert logs[0]["stage"] == "EXTRACT"
    assert logs[0]["payload"] == {"facts": ["a"]}


def test_record_operation(tmp_storage):
    c = CacheStore(tmp_storage)
    c.record_operation(op="ADD", node_id="n1", user_id="u")


def test_history_append_and_list(tmp_storage):
    h = HistoryStore(tmp_storage, persist=True)
    h.append(event="ADD", node_id="n1", user_id="u", old=None, new={"content": "x"})
    h.append(
        event="SUPERSEDE",
        node_id="n1",
        user_id="u",
        old={"content": "x"},
        new={"content": "y"},
    )
    rows = h.list_for_node("n1")
    assert len(rows) == 2
    assert rows[0]["event"] == "ADD"
    assert rows[0]["old"] is None
    assert rows[1]["new"] == {"content": "y"}


def test_history_disabled_is_noop(tmp_storage):
    h = HistoryStore(tmp_storage, persist=False)
    h.append(event="ADD", node_id="n1", user_id="u", old=None, new={"content": "x"})
    assert h.list_for_node("n1") == []


def test_purge_done_queues(tmp_storage):
    c = CacheStore(tmp_storage)
    c.enqueue_reconcile_task(app_id="app", user_id="u", agent_id="", node_ids=["a"])
    task = c.dequeue_reconcile_task(app_id="app", user_id="u")
    assert task is not None
    assert c.purge_done_queues() == 1
    row = c.conn.execute("SELECT COUNT(*) AS n FROM reconcile_queue").fetchone()
    assert int(row["n"]) == 0
