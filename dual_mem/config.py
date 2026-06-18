# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runtime configuration via pydantic-settings; resolves Settings from init
args, DUAL_MEM_* env vars and a YAML file, and derives mode-based flags.
"""
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

    mode: Literal["lite", "pro", "ultra"] = "pro"
    storage_dir: str = "./.dual_mem_data"

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key: str = ""
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536

    auth_disabled: bool = True
    app_whitelist: list[str] = ["default"]

    system2_trigger_mode: Literal["per_write", "manual", "scheduled"] = "per_write"

    # System2 聚类的相似度阈值（cosine）。低于该相似度的事实不归为一簇。
    # 默认值适配出厂的 text-embedding-3-small（其同主题事实 cosine 普遍偏低）；
    # 若改用相似度更集中的中文 embedding（如 bge/Qwen），可调高到 0.55/0.75。
    cluster_stage1_sim: float = 0.42
    cluster_stage2_sim: float = 0.55

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
    def agent_mode(self) -> str:
        """Derived LLM-cognition mode: disabled for lite, full otherwise."""
        return "disabled" if self.mode == "lite" else "full"

    @property
    def enable_graph(self) -> bool:
        """Whether the graph store is enabled (ultra mode only)."""
        return self.mode == "ultra"
