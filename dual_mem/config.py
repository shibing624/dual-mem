# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runtime configuration via pydantic-settings; resolves Settings from init
args, DUAL_MEM_* env vars and a YAML file, and derives mode-based flags.
"""
from __future__ import annotations

import logging
import os
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

logger = logging.getLogger("dual_mem.config")

# Legacy alias mapping for users who still pass pro/ultra (or the older emb/lite). The
# embedding-only / no-LLM ``emb`` (a.k.a. ``lite``) mode has been removed; passing it now
# raises so users notice and supply LLM credentials instead of getting a silent downgrade.
_MODE_ALIASES = {"pro": "system1", "ultra": "dual"}
_REMOVED_MODES = {"lite", "emb"}

DEFAULT_CONFIG_PATH = Path.home() / ".dual_mem" / "config.yaml"


def config_path() -> Path:
    """Resolve the YAML config path, honoring the DUAL_MEM_CONFIG_FILE override."""
    override = os.environ.get("DUAL_MEM_CONFIG_FILE")
    return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH


def _default_config_template() -> str:
    """Load the bundled default YAML shipped inside the dual_mem package."""
    return resources.files("dual_mem").joinpath("config.default.yaml").read_text(encoding="utf-8")


def ensure_config_file() -> Path:
    """Create ``~/.dual_mem/config.yaml`` with defaults when missing (MetaGPT-style bootstrap).

    Skipped when ``DUAL_MEM_CONFIG_FILE`` points at a custom path — only the default
    home location is auto-created so tests and explicit overrides stay predictable.
    """
    if os.environ.get("DUAL_MEM_CONFIG_FILE"):
        return config_path()
    path = DEFAULT_CONFIG_PATH
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_default_config_template(), encoding="utf-8")
    logger.info(
        "Created default config at %s — edit llm_api_key and embed_api_key, "
        "or set DUAL_MEM_LLM_API_KEY / DUAL_MEM_EMBED_API_KEY.",
        path,
    )
    return path


def ensure_storage_dir(storage_dir: str) -> Path:
    """Create the on-disk data root if missing (SQLite/Chroma/Kuzu need the parent dir)."""
    path = Path(storage_dir).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_app_id(settings: Settings, app_id: str | None) -> str:
    """Return explicit ``app_id`` or ``settings.default_app_id``."""
    return app_id if app_id is not None else settings.default_app_id


def resolve_app_ids(settings: Settings, app_ids: list[str] | None) -> list[str]:
    """Return explicit ``app_ids`` or ``[settings.default_app_id]``."""
    return app_ids if app_ids is not None else [settings.default_app_id]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DUAL_MEM_",
        extra="ignore",
    )

    # Mode (paper-aligned naming):
    #   system1 — synchronous System1 cognition: gate -> LLM extract -> fast-write into
    #             L0-L4 (was "pro"). This is the default.
    #   dual    — System1 + asynchronous System2 distillation/graph (L6 schema / L7
    #             intention, reconcile worker, ReAct agent; was "ultra"). The full
    #             dual-system pipeline that gives the SDK its name.
    # Both modes require an LLM API key and an embedding API key; missing credentials
    # cause MemoryClient to raise on construction.
    mode: Literal["system1", "dual"] = "system1"
    storage_dir: str = "./.dual_mem_data"

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # Use OpenAI JSON mode (response_format=json_object) for JSON-returning calls.
    # Turn off only if the endpoint does not support response_format.
    llm_json_mode: bool = True

    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key: str = ""
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536

    auth_disabled: bool = True
    app_whitelist: list[str] = ["default"]
    # Default tenant namespace when add/list/search omit app_id (single-product default).
    default_app_id: str = "default"

    system2_trigger_mode: Literal["per_write", "manual", "scheduled"] = "per_write"
    # Scheduled-mode background loop period in seconds (only used when trigger_mode=scheduled).
    system2_schedule_interval_sec: int = 300

    # System2 聚类的相似度阈值（cosine）。低于该相似度的事实不归为一簇。
    cluster_stage1_sim: float = 0.42
    cluster_stage2_sim: float = 0.55

    # Reconciler 额外的 LLM 召回查询改写：默认关闭（语义召回本身够用，省一次 LLM/写入）。
    reconcile_search_query: bool = False
    # Run reconciler synchronously inside the write path (strong consistency for evolution
    # chains; also raises latency to ~2 LLM calls per add). Default off: async reconcile
    # via ReconcilerWorker (per_write task drains chains within seconds).
    reconcile_sync: bool = False
    # System2 ReAct loop iteration cap (the agent stops earlier when the LLM emits no more
    # tool_calls). The legacy single-shot ops JSON path has been removed.
    system2_max_iters: int = 10

    # Attentional gate: skip extraction for low-value content (pleasantries, no novelty).
    gate_enabled: bool = True
    gate_threshold: float = 0.3

    # Embedding write-side batching window for embed_queued (does not affect search-side embed).
    embed_queue_batch_size: int = 32
    embed_queue_window_ms: float = 200.0
    # Merge L1 dialogue + Gate user-turn embeds into one API call on add(messages=...).
    # Saves ~1 RTT + queue window; bypasses embed_queued coalescing — keep false for
    # high-concurrency write throughput (default).
    embed_merge_l1_gate: bool = False

    # Cross-domain Sweeper: behavior abstraction + cosine collision + Union-Find induction.
    cross_domain_enable: bool = False
    cross_domain_min_basics: int = 5
    cross_domain_threshold: float = 0.7

    # V2 read pipeline: hybrid (default) = QueryUnderstanding -> AnchorSearch (5 paths) ->
    # GraphExpander -> FusionScorer; legacy = original three-route + bm25 RRF rerank baseline.
    reader_mode: Literal["hybrid", "legacy"] = "hybrid"

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, v):
        """Map legacy pro/ultra to system1/dual; reject removed emb/lite modes."""
        if isinstance(v, str):
            key = v.strip().lower()
            if key in _REMOVED_MODES:
                raise ValueError(
                    f"Settings.mode={v!r} (embedding-only / no-LLM mode) has been removed. "
                    "dual-mem now requires both LLM and embedding API keys; "
                    "use mode='system1' (default) or mode='dual'."
                )
            if key in _MODE_ALIASES:
                canonical = _MODE_ALIASES[key]
                logger.warning(
                    "Settings.mode=%r is deprecated; use %r instead.",
                    v,
                    canonical,
                )
                return canonical
        return v

    @field_validator("app_whitelist", mode="before")
    @classmethod
    def _split_whitelist(cls, v):
        """Accept a comma-separated string for app_whitelist and split it into a list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order config sources as init args > env vars > YAML file."""
        ensure_config_file()
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=config_path())
        return (init_settings, env_settings, yaml_source)

    @property
    def enable_graph(self) -> bool:
        """Whether the graph store is enabled (``dual`` mode only)."""
        return self.mode == "dual"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Settings:
        """Build ``Settings`` from a mem0/Hy-style nested dict (flat keys also accepted).

        Nested sections:

        - ``llm``: ``api_key`` / ``base_url`` / ``model`` → ``llm_*``
        - ``embedder`` or ``embed``: same → ``embed_*``
        - ``vector_store.persist_directory`` → ``storage_dir`` (when set)

        Unknown top-level keys are ignored (``extra="ignore"``).
        """
        flat = _flatten_config_dict(config_dict)
        return cls(**flat)


def _flatten_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Map mem0/Hy nested provider blocks onto flat ``Settings`` field names."""
    flat = dict(config_dict)

    llm = flat.pop("llm", None)
    if isinstance(llm, dict):
        _map_provider_section(llm, flat, prefix="llm")

    embed_section = flat.pop("embedder", None)
    if embed_section is None:
        embed_section = flat.pop("embed", None)
    if isinstance(embed_section, dict):
        _map_provider_section(embed_section, flat, prefix="embed")

    vector_store = flat.pop("vector_store", None)
    if isinstance(vector_store, dict):
        persist = vector_store.get("persist_directory")
        if persist:
            flat.setdefault("storage_dir", persist)
        dims = vector_store.get("embedding_dims")
        if dims is not None:
            flat.setdefault("embed_dim", dims)

    for drop_key in (
        "graph_store",
        "enable_graph",
        "cache",
        "history_store",
        "providers",
    ):
        flat.pop(drop_key, None)

    return flat


def _map_provider_section(section: dict[str, Any], flat: dict[str, Any], *, prefix: str) -> None:
    """Copy ``api_key`` / ``base_url`` / ``model`` into ``{prefix}_*`` Settings keys."""
    mapping = {
        "api_key": f"{prefix}_api_key",
        "base_url": f"{prefix}_base_url",
        "model": f"{prefix}_model",
    }
    for src, dst in mapping.items():
        val = section.get(src)
        if val is not None and val != "":
            flat.setdefault(dst, val)
