# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runtime configuration via pydantic-settings; resolves Settings from init
args, DUAL_MEM_* env vars and a YAML file, and derives mode-based flags.
"""
import logging
import os
from pathlib import Path
from typing import Literal

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
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=config_path())
        return (init_settings, env_settings, yaml_source)

    @property
    def enable_graph(self) -> bool:
        """Whether the graph store is enabled (``dual`` mode only)."""
        return self.mode == "dual"
