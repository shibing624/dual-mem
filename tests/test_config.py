import pytest

from dual_mem.config import Settings


def test_defaults():
    s = Settings()
    assert s.mode == "pro"
    assert s.embed_dim == 1536
    assert s.auth_disabled is True
    assert s.system2_trigger_mode == "per_write"


def test_app_whitelist_comma_string():
    s = Settings(app_whitelist="a, b ,c")
    assert s.app_whitelist == ["a", "b", "c"]


def test_app_whitelist_list():
    s = Settings(app_whitelist=["x", "y"])
    assert s.app_whitelist == ["x", "y"]


def test_agent_mode_derived():
    assert Settings(mode="lite").agent_mode == "disabled"
    assert Settings(mode="pro").agent_mode == "full"
    assert Settings(mode="ultra").agent_mode == "full"


def test_enable_graph_derived():
    assert Settings(mode="lite").enable_graph is False
    assert Settings(mode="pro").enable_graph is False
    assert Settings(mode="ultra").enable_graph is True


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        Settings(mode="turbo")


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("DUAL_MEM_MODE", "ultra")
    monkeypatch.setenv("DUAL_MEM_EMBED_DIM", "768")
    s = Settings()
    assert s.mode == "ultra"
    assert s.embed_dim == 768
