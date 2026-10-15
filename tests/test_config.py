import pytest

from dual_mem.config import (
    CHARS_PER_TOKEN,
    EMBED_MAX_TOKENS,
    LLM_COMPLETION_RESERVE,
    LLM_CONTEXT_WINDOW,
    Settings,
)


def test_defaults():
    s = Settings()
    assert s.mode == "system1"
    assert s.embed_dim == 1536
    assert s.llm_context_window == LLM_CONTEXT_WINDOW
    assert s.llm_completion_reserve == LLM_COMPLETION_RESERVE
    assert s.chars_per_token == CHARS_PER_TOKEN
    assert s.embed_max_tokens == EMBED_MAX_TOKENS
    assert s.llm_input_max_chars == int(
        (LLM_CONTEXT_WINDOW - LLM_COMPLETION_RESERVE) * CHARS_PER_TOKEN
    )
    assert s.embed_input_max_chars == int(EMBED_MAX_TOKENS * CHARS_PER_TOKEN)
    assert s.auth_disabled is True
    assert s.default_app_id == "default"
    assert s.reconcile_concurrency == 4
    assert s.persist_history is False
    assert s.purge_done_queues is True


def test_app_whitelist_comma_string():
    s = Settings(app_whitelist="a, b ,c")
    assert s.app_whitelist == ["a", "b", "c"]


def test_app_whitelist_list():
    s = Settings(app_whitelist=["x", "y"])
    assert s.app_whitelist == ["x", "y"]


def test_enable_graph_derived():
    assert Settings(mode="system1").enable_graph is False
    assert Settings(mode="dual").enable_graph is True


@pytest.mark.parametrize("bad_mode", ["turbo", "pro", "ultra", "emb", "lite"])
def test_invalid_mode_raises(bad_mode):
    with pytest.raises(ValueError):
        Settings(mode=bad_mode)


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


def test_preset_default_equals_code_defaults():
    s = Settings()
    assert s.preset == "default"
    # default preset is empty → these stay at their conservative code defaults
    assert s.reconcile_concurrency == 4
    assert s.reconcile_candidate_limit == 64


def test_preset_high_recall_fills_knobs():
    s = Settings(preset="high_recall")
    assert s.reconcile_candidate_limit == 128
    # latency knobs are orthogonal — NOT touched by any preset


def test_explicit_field_overrides_preset():
    # init arg beats preset
    s = Settings(preset="high_recall", reconcile_candidate_limit=32)
    assert s.reconcile_candidate_limit == 32


def test_env_field_overrides_preset(monkeypatch):
    monkeypatch.setenv("DUAL_MEM_PRESET", "high_recall")
    monkeypatch.setenv("DUAL_MEM_RECONCILE_CANDIDATE_LIMIT", "32")
    s = Settings()
    assert s.preset == "high_recall"
    assert s.reconcile_candidate_limit == 32  # env wins over preset


def test_preset_selected_from_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("preset: high_recall\n", encoding="utf-8")
    monkeypatch.setenv("DUAL_MEM_CONFIG_FILE", str(cfg))
    s = Settings()
    assert s.preset == "high_recall"
    assert s.reconcile_candidate_limit == 128


@pytest.mark.parametrize("bad_preset", ["fast", "prod", "hi-recall"])
def test_invalid_preset_raises(bad_preset):
    with pytest.raises(ValueError):
        Settings(preset=bad_preset)


def test_missing_jev_credentials_fails_before_network(
    tmp_storage, fake_embed, fake_llm, monkeypatch
):
    from dual_mem import MemoryClient
    from dual_mem.client import MissingCredentialsError

    monkeypatch.delenv("DUAL_MEM_JEV_API_KEY", raising=False)
    with pytest.raises(MissingCredentialsError, match="Jev requires"):
        MemoryClient(
            settings=Settings(storage_dir=tmp_storage, jev_api_key=""),
            embed=fake_embed,
            llm=fake_llm,
        )


def test_flat_jev_config():
    s = Settings.from_dict(
        {"jev_api_key": "test-jev", "jev_endpoint": "https://example.test/v1/systemone"}
    )
    assert s.jev_api_key == "test-jev" and s.jev_endpoint.endswith("/v1/systemone")


@pytest.mark.parametrize(
    "key",
    [
        "reconcile_sync",
        "reconcile_skip_llm",
        "reconcile_weak_candidate_score",
        "llm",
        "embedder",
        "vector_store",
        "jev",
    ],
)
def test_removed_controls_and_nested_config_are_rejected(key):
    with pytest.raises(ValueError):
        Settings.from_dict({key: True})


def test_old_yaml_controls_are_rejected(tmp_path, monkeypatch):
    cfg = tmp_path / "old-config.yaml"
    cfg.write_text("reconcile_sync: true\n")
    monkeypatch.setenv("DUAL_MEM_CONFIG_FILE", str(cfg))
    with pytest.raises(ValueError):
        Settings()
