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

DEFAULT_CONFIG_PATH = Path.home() / ".dual_mem" / "config.yaml"


def config_path() -> Path:
    """Resolve the YAML config path, honoring the DUAL_MEM_CONFIG_FILE override."""
    override = os.environ.get("DUAL_MEM_CONFIG_FILE")
    return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH


def _default_config_template() -> str:
    """Load the bundled default YAML shipped inside the dual_mem package."""
    return (
        resources.files("dual_mem")
        .joinpath("config.default.yaml")
        .read_text(encoding="utf-8")
    )


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
        "Created default config at %s — edit jev_api_key, llm_api_key and embed_api_key, "
        "or set DUAL_MEM_JEV_API_KEY / DUAL_MEM_LLM_API_KEY / DUAL_MEM_EMBED_API_KEY.",
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


# Token→char estimate shared by LLM prompt budgeting and embed chunk sizing.
CHARS_PER_TOKEN = 2.5
LLM_CHARS_PER_TOKEN = CHARS_PER_TOKEN

# LLM context budgeting — override via YAML or DUAL_MEM_LLM_CONTEXT_WINDOW etc.
LLM_CONTEXT_WINDOW = 32768
LLM_COMPLETION_RESERVE = 4096

# Embed input budget (tokens per API call; longer inputs are head-truncated).
EMBED_MAX_TOKENS = 8000
EMBED_RETRY_ATTEMPTS = 3
EMBED_RETRY_BASE_DELAY = 0.5


# Presets change only material budget. Jev remains authoritative in every preset.
PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "high_recall": {"reconcile_candidate_limit": 128},
}
PRESET_NAMES = tuple(PRESETS.keys())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DUAL_MEM_",
        extra="ignore",
    )

    # Both modes append System1 facts/profiles using Jev decisions.
    # dual adds explicit digest for cross-evidence L3 cognition.
    mode: Literal["system1", "dual"] = "system1"
    storage_dir: str = "./.dual_mem_data"

    # Tuning preset: pick ONE named scenario instead of hand-tuning the advanced knobs
    # below. The preset only fills in fields you did NOT set explicitly (init/env/YAML
    # override it). See ``PRESETS`` for what each one changes.
    #   default     — balanced general SDK usage (== code defaults).
    #   high_recall — expand Jev material budget; preserve normal relation decisions.
    preset: Literal["default", "high_recall"] = "default"

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # Use OpenAI JSON mode (response_format=json_object) for JSON-returning calls.
    # Turn off only if the endpoint does not support response_format.
    llm_json_mode: bool = True
    # Provider-specific body (Volces thinking depth, vendor extensions, etc.).
    extra_body: dict[str, Any] = {}
    # Provider-specific headers (internal platform auth, tracing, etc.).
    extra_headers: dict[str, str] = {}
    llm_timeout: int = 60
    # Model context window (total tokens). Used with completion_reserve to derive
    # per-call char budget; long prompts are split into chunks + merged (not truncated).
    llm_context_window: int = LLM_CONTEXT_WINDOW
    llm_completion_reserve: int = LLM_COMPLETION_RESERVE
    chars_per_token: float = CHARS_PER_TOKEN

    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key: str = ""
    embed_model: str = "text-embedding-3-small"
    # Vector dimension — set to your embedding model's native output size (e.g. 1536, 1024).
    embed_dim: int = 1536
    embed_timeout: int = 30
    # Max input tokens per embed API call; longer texts are head-truncated (L1 anchors only).
    embed_max_tokens: int = EMBED_MAX_TOKENS
    embed_retry_attempts: int = EMBED_RETRY_ATTEMPTS
    embed_retry_base_delay: float = EMBED_RETRY_BASE_DELAY

    auth_disabled: bool = True
    app_whitelist: list[str] = ["default"]
    # Default tenant namespace when add/list/search omit app_id (single-product default).
    default_app_id: str = "default"

    # Optional full serialization; default overlaps model I/O and validates scope CAS.
    write_serialize_per_user: bool = False

    # Typed decision service (official TypeSafe or Venus-compatible endpoint).
    jev_endpoint: str = "https://api.typesafe.ai/v1/systemone"
    jev_api_key: str = ""
    jev_model: str = "jev-1.13.0"
    jev_timeout: float = 30.0
    jev_retry_attempts: int = 3
    jev_max_questions: int = 64
    jev_max_request_chars: int = 100_000
    jev_max_question_chars: int = 60_000
    reconcile_concurrency: int = 4
    reconcile_candidate_limit: int = 64
    reconcile_batch_size: int = 12
    reconcile_retry_attempts: int = 3
    system2_max_prompt_facts: int = 120

    # Legacy generic extractor cap in tokens (0 = disabled → rely on the LLM chunk+merge path).
    # Applies to the final content of BOTH content= and messages= (after history shaping). This
    # is the last-resort size cap (e.g. when user turns alone are huge); it is layered ON TOP of
    # — and does not conflict with — the role-aware history shaping below.
    extract_max_content_tokens: int = 0
    # Retry once on empty/unparseable JSON (temperature=0 + JSON-only reinforcement prompt).
    extract_retry_on_failure: bool = True
    # Few-shot 示例：extract prompt 末尾追加示例，引导 4B 模型稳定格式。
    extract_few_shot_enabled: bool = False
    # Multi-turn extract input shaping (messages=...): no user/assistant turn is ever dropped.
    # When the dialogue exceeds ``extract_dialogue_context_ratio`` of ``llm_context_window``,
    # only assistant turns are truncated to ``extract_assistant_max_tokens``; user turns stay
    # full (primary memory signal). role=system dialogue turns are ignored (not memory).
    # The extract LLM's EXTRACT_* template system prompt is instruction-only, not chat history.
    extract_dialogue_context_ratio: float = 0.7
    extract_assistant_max_tokens: int = 512

    # Embedding write-side batching window for embed_queued (does not affect search-side embed).
    embed_queue_batch_size: int = 32
    embed_queue_window_ms: float = 200.0
    embed_cache_size: int = 10_000
    # Append-only history.db audit log (ADD/SUPERSEDE/DELETE snapshots). Off by default —
    # not used on the read path; enable for compliance / debugging.
    persist_history: bool = False
    # Drop reconcile/s2 queue rows after digest drains them (keeps cache.db small).
    purge_done_queues: bool = True

    # Hybrid read fusion tunables (semantic vs in-pool BM25 rerank weight; graph L6 evidence
    # boost). Sum of the two weights need not be 1; evidence boost saturates at the given count.
    hybrid_w_sem: float = 0.6
    hybrid_w_bm25: float = 0.4
    normal_candidate_multiplier: float = 3.0
    hybrid_evidence_boost_max: float = 0.3
    hybrid_evidence_saturate: int = 5
    # Coding/tool-use memory subsystem — separate write/store path for engineering
    # conversations containing tool calls. When enabled, add() first checks for tool
    # messages; if found, Jev decides coding vs chat, and coding memories
    # go to a separate SQLite+VDB store with task/search_keys/solution schema.
    coding_enabled: bool = False  # opt-in; experimental
    coding_db_path: str = ""  # empty = auto: {storage_dir}/coding_memory.db
    coding_tool_result_max_bytes: int = 2048

    @field_validator(
        "jev_timeout",
        "jev_retry_attempts",
        "jev_max_questions",
        "jev_max_request_chars",
        "jev_max_question_chars",
        "reconcile_concurrency",
        "reconcile_candidate_limit",
        "reconcile_batch_size",
        "reconcile_retry_attempts",
        "system2_max_prompt_facts",
    )
    @classmethod
    def _positive_decision_limit(cls, value: int | float) -> int | float:
        if value <= 0:
            raise ValueError("decision and cognition limits must be positive")
        return value

    @field_validator("reconcile_candidate_limit")
    @classmethod
    def _choice_material_limit(cls, value: int) -> int:
        if value > 220:
            raise ValueError("candidate limit must be <=220 to fit Jev choice options")
        return value

    @field_validator("reconcile_batch_size")
    @classmethod
    def _proposal_limit(cls, value: int) -> int:
        if value > 24:
            raise ValueError("batch size must be <=24")
        return value

    @field_validator("app_whitelist", mode="before")
    @classmethod
    def _split_whitelist(cls, v):
        """Accept a comma-separated string for app_whitelist and split it into a list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("llm_context_window")
    @classmethod
    def _llm_context_window_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("llm_context_window must be positive")
        return v

    @field_validator("llm_completion_reserve")
    @classmethod
    def _llm_completion_reserve_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("llm_completion_reserve must be >= 0")
        return v

    @field_validator("chars_per_token")
    @classmethod
    def _chars_per_token_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("chars_per_token must be positive")
        return v

    @field_validator("embed_max_tokens")
    @classmethod
    def _embed_max_tokens_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embed_max_tokens must be positive")
        return v

    @property
    def llm_input_max_chars(self) -> int:
        """Estimated prompt char budget from context window minus completion reserve."""
        usable_tokens = self.llm_context_window - self.llm_completion_reserve
        if usable_tokens <= 0:
            return 0
        return int(usable_tokens * self.chars_per_token)

    @property
    def embed_input_max_chars(self) -> int:
        """Char chunk size derived from ``embed_max_tokens`` × ``chars_per_token``."""
        return int(self.embed_max_tokens * self.chars_per_token)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order config sources as init args > env vars > YAML file > preset > code defaults.

        The preset source sits at the LOWEST precedence (just above hard-coded field
        defaults): it only fills fields the user left unset via init/env/YAML. The preset
        NAME itself is resolved from those same three sources (in that order) so users can
        select a preset from anywhere.
        """
        ensure_config_file()
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=config_path())
        preset_source = _PresetSettingsSource(
            settings_cls, init_settings, env_settings, yaml_source
        )
        return (init_settings, env_settings, yaml_source, preset_source)

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


class _PresetSettingsSource(PydanticBaseSettingsSource):
    """Lowest-precedence source that injects a tuning preset's field values.

    The preset NAME is resolved from the higher-precedence sources (init > env > YAML);
    the resolved preset's dict is then returned so pydantic-settings treats those values
    as defaults — any field the user set explicitly elsewhere still wins.
    """

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        init_source: PydanticBaseSettingsSource,
        env_source: PydanticBaseSettingsSource,
        yaml_source: PydanticBaseSettingsSource,
    ) -> None:
        super().__init__(settings_cls)
        self._sources = (init_source, env_source, yaml_source)

    def _resolve_preset_name(self) -> str:
        """Read ``preset`` from init/env/YAML in precedence order; fall back to default."""
        for source in self._sources:
            try:
                values = source()
            except Exception:
                continue
            name = values.get("preset")
            if name:
                return str(name)
        return "default"

    def get_field_value(self, field, field_name):  # pragma: no cover - interface stub
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        name = self._resolve_preset_name()
        preset = PRESETS.get(name)
        if preset is None:
            valid = ", ".join(PRESET_NAMES)
            raise ValueError(f"Unknown preset {name!r}. Valid presets: {valid}.")
        return dict(preset)


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

    jev = flat.pop("jev", None)
    if isinstance(jev, dict):
        for key, value in jev.items():
            flat.setdefault(f"jev_{key}", value)

    vector_store = flat.pop("vector_store", None)
    if isinstance(vector_store, dict):
        vector_config = vector_store.get("config")
        if isinstance(vector_config, dict):
            vector_store = {**vector_config, **vector_store}
        persist = vector_store.get("persist_directory")
        if persist:
            flat.setdefault("storage_dir", persist)
        dims = vector_store.get("embedding_dims")
        if dims is None:
            dims = vector_store.get("embedding_model_dims")
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


def _map_provider_section(
    section: dict[str, Any], flat: dict[str, Any], *, prefix: str
) -> None:
    """Copy provider keys into ``{prefix}_*`` Settings fields."""
    provider_config = section.get("config")
    if isinstance(provider_config, dict):
        section = {**provider_config, **section}

    mapping = {
        "api_key": f"{prefix}_api_key",
        "base_url": f"{prefix}_base_url",
        "model": f"{prefix}_model",
        "json_mode": f"{prefix}_json_mode",
        "extra_body": "extra_body",
        "timeout": f"{prefix}_timeout",
        "retry_attempts": f"{prefix}_retry_attempts",
        "retry_base_delay": f"{prefix}_retry_base_delay",
        "chars_per_token": "chars_per_token",
    }
    if prefix == "llm":
        mapping["context_window"] = "llm_context_window"
        mapping["max_model_len"] = "llm_context_window"
        mapping["completion_reserve"] = "llm_completion_reserve"
        mapping["extra_headers"] = "extra_headers"
    if prefix == "embed":
        mapping["max_tokens"] = "embed_max_tokens"
        mapping["embedding_dims"] = "embed_dim"
    for src, dst in mapping.items():
        val = section.get(src)
        if val is not None and val != "":
            flat.setdefault(dst, val)
