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


def test_ensure_config_file_creates_default(tmp_path, monkeypatch):
    """Default ~/.dual_mem/config.yaml is bootstrapped when missing."""
    monkeypatch.delenv("DUAL_MEM_CONFIG_FILE", raising=False)
    monkeypatch.setattr("dual_mem.config.DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
    from dual_mem.config import ensure_config_file

    path = ensure_config_file()
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "llm_api_key" in text
    assert "embed_api_key" in text
    assert path.read_text(encoding="utf-8") == text  # idempotent
    ensure_config_file()
    assert path.read_text(encoding="utf-8") == text


def test_ensure_storage_dir_creates_nested(tmp_path):
    from dual_mem.config import ensure_storage_dir

    target = tmp_path / "nested" / "dual_mem_data"
    ensure_storage_dir(str(target))
    assert target.is_dir()
