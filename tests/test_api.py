import pytest
from fastapi.testclient import TestClient

from dual_mem import MemoryClient
from dual_mem.api import create_app
from dual_mem.config import Settings


@pytest.fixture
def app_client(tmp_storage, fake_embed):
    settings = Settings(storage_dir=tmp_storage, mode="lite", auth_disabled=True)
    client = MemoryClient(settings=settings, embed=fake_embed)
    app = create_app(client=client, settings=settings)
    return TestClient(app)


def test_add_search_get_list_delete_flow(app_client):
    added = app_client.post(
        "/v1/memories/",
        json={"content": "用户喜欢喝咖啡", "app_id": "app", "user_id": "u"},
    )
    assert added.status_code == 200
    body = added.json()
    assert body["success"] is True
    memory_id = body["memory_id"]

    searched = app_client.post(
        "/v1/memories/search",
        json={"query": "用户喜欢喝咖啡", "app_ids": ["app"], "user_id": "u", "min_score": 0.4},
    )
    assert searched.status_code == 200
    memories = searched.json()["memories"]
    assert set(memories.keys()) == {"profile", "proactive", "normal"}
    assert any(m["memory_id"] == memory_id for m in memories["normal"])

    got = app_client.get(f"/v1/memories/{memory_id}")
    assert got.status_code == 200
    assert got.json()["content"] == "用户喜欢喝咖啡"

    listed = app_client.get("/v1/memories/", params={"app_id": "app", "user_id": "u"})
    assert listed.status_code == 200
    assert any(m["memory_id"] == memory_id for m in listed.json())

    deleted = app_client.delete(f"/v1/memories/{memory_id}")
    assert deleted.status_code == 200
    assert deleted.json()["success"] is True


def test_add_requires_content_or_messages(app_client):
    resp = app_client.post(
        "/v1/memories/", json={"app_id": "app", "user_id": "u"}
    )
    assert resp.status_code == 400
    assert resp.json()["error_code"] == 400


def test_delete_missing_404(app_client):
    resp = app_client.delete("/v1/memories/no-such-id")
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == 404


def test_get_missing_404(app_client):
    resp = app_client.get("/v1/memories/no-such-id")
    assert resp.status_code == 404
    assert resp.json()["error_code"] == 404


def test_delete_bulk_requires_confirm(app_client):
    app_client.post(
        "/v1/memories/", json={"content": "批量1", "app_id": "app", "user_id": "u"}
    )
    no_confirm = app_client.delete("/v1/memories/", params={"app_id": "app", "user_id": "u"})
    assert no_confirm.status_code == 400
    assert no_confirm.json()["error_code"] == 400

    done = app_client.delete(
        "/v1/memories/", params={"app_id": "app", "user_id": "u", "confirm": "true"}
    )
    assert done.status_code == 200
    assert done.json()["success"] is True


def test_health_ping_info(app_client):
    health = app_client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "SERVING"

    ping = app_client.get("/ping")
    assert ping.status_code == 200
    assert ping.json()["message"] == "pong"

    info = app_client.get("/info")
    assert info.status_code == 200
    assert info.json()["mode"] == "lite"
    assert "sdk_version" in info.json()


@pytest.fixture
def auth_client(tmp_storage, fake_embed):
    settings = Settings(
        storage_dir=tmp_storage,
        mode="lite",
        auth_disabled=False,
        app_whitelist=["allowed"],
    )
    client = MemoryClient(settings=settings, embed=fake_embed)
    app = create_app(client=client, settings=settings)
    return TestClient(app)


def test_auth_missing_token_403(auth_client):
    resp = auth_client.post(
        "/v1/memories/",
        json={"content": "x", "app_id": "allowed", "user_id": "u"},
    )
    assert resp.status_code == 403
    assert resp.json()["error_code"] == 403


def test_auth_app_not_whitelisted_403(auth_client):
    resp = auth_client.post(
        "/v1/memories/",
        headers={"Authorization": "Bearer sk-test"},
        json={"content": "x", "app_id": "stranger", "user_id": "u"},
    )
    assert resp.status_code == 403
    assert resp.json()["error_code"] == 403


def test_auth_search_app_not_whitelisted_403(auth_client):
    resp = auth_client.post(
        "/v1/memories/search",
        headers={"Authorization": "Bearer sk-test"},
        json={"query": "q", "app_ids": ["stranger"], "user_id": "u"},
    )
    assert resp.status_code == 403
    assert resp.json()["error_code"] == 403


def test_auth_allowed_passes(auth_client):
    resp = auth_client.post(
        "/v1/memories/",
        headers={"Authorization": "Bearer sk-test"},
        json={"content": "ok", "app_id": "allowed", "user_id": "u"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
