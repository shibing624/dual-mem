import pytest

from dual_mem.config import Settings


def test_defaults():
    s = Settings()
    assert s.mode == "system1"
    assert s.embed_dim == 1536
    assert s.auth_disabled is True
    assert s.system2_trigger_mode == "per_write"


def test_app_whitelist_comma_string():
    s = Settings(app_whitelist="a, b ,c")
    assert s.app_whitelist == ["a", "b", "c"]


def test_app_whitelist_list():
    s = Settings(app_whitelist=["x", "y"])
    assert s.app_whitelist == ["x", "y"]


def test_enable_graph_derived():
    assert Settings(mode="system1").enable_graph is False
    assert Settings(mode="dual").enable_graph is True


def test_mode_aliases(caplog):
    """Legacy pro/ultra aliases are still accepted with a deprecation warning."""
    import logging

    with caplog.at_level(logging.WARNING, logger="dual_mem.config"):
        assert Settings(mode="pro").mode == "system1"
        assert Settings(mode="ultra").mode == "dual"
    assert any("deprecated" in r.message for r in caplog.records)


def test_emb_mode_removed():
    """The embedding-only modes (emb / lite) have been removed; constructing them must raise."""
    with pytest.raises(ValueError, match="removed"):
        Settings(mode="emb")
    with pytest.raises(ValueError, match="removed"):
        Settings(mode="lite")


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        Settings(mode="turbo")


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("DUAL_MEM_MODE", "dual")
    monkeypatch.setenv("DUAL_MEM_EMBED_DIM", "768")
    s = Settings()
    assert s.mode == "dual"
    assert s.embed_dim == 768


def test_yaml_source(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: dual\nembed_dim: 256\nllm_model: my-model\napp_whitelist:\n  - alpha\n  - beta\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DUAL_MEM_CONFIG_FILE", str(cfg))
    s = Settings()
    assert s.mode == "dual"
    assert s.embed_dim == 256
    assert s.llm_model == "my-model"
    assert s.app_whitelist == ["alpha", "beta"]


def test_env_overrides_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mode: dual\n", encoding="utf-8")
    monkeypatch.setenv("DUAL_MEM_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("DUAL_MEM_MODE", "system1")
    assert Settings().mode == "system1"


def test_init_overrides_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mode: dual\n", encoding="utf-8")
    monkeypatch.setenv("DUAL_MEM_CONFIG_FILE", str(cfg))
    assert Settings(mode="system1").mode == "system1"
