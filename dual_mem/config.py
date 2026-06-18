from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DUAL_MEM_",
        env_file=".env",
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

    @field_validator("app_whitelist", mode="before")
    @classmethod
    def _split_whitelist(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def agent_mode(self) -> str:
        return "disabled" if self.mode == "lite" else "full"

    @property
    def enable_graph(self) -> bool:
        return self.mode == "ultra"
