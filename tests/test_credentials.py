"""Tests for credential validation in MemoryClient.

dual-mem requires both LLM and embedding API keys; missing credentials raise on construction
(no silent embedding-only fallback). Injected ``embed=`` / ``llm=`` clients bypass the api_key
check so tests / custom backends can plug in directly.
"""
import pytest

from dual_mem.client import MemoryClient, MissingCredentialsError
from dual_mem.config import Settings

from conftest import FakeLLMClient


def test_injected_clients_bypass_credential_check(tmp_storage, fake_embed):
    """When both embed and llm are injected, missing api_key strings are tolerated."""
    settings = Settings(
        storage_dir=tmp_storage, mode="dual", llm_api_key="", embed_api_key=""
    )
    client = MemoryClient(
        settings=settings, embed=fake_embed, llm=FakeLLMClient(responses={})
    )
    assert client.settings.mode == "dual"
    assert client.factory.has_user_llm is True


def test_missing_llm_key_raises(tmp_storage, fake_embed):
    """Without an LLM api key (and no injected llm), construction must fail fast."""
    settings = Settings(
        storage_dir=tmp_storage,
        mode="system1",
        llm_api_key="",
        embed_api_key="present",
    )
    with pytest.raises(MissingCredentialsError, match="llm_api_key"):
        MemoryClient(settings=settings, embed=fake_embed)


def test_missing_embed_key_raises(tmp_storage):
    """Without an embedding api key (and no injected embed), construction must fail fast."""
    settings = Settings(
        storage_dir=tmp_storage,
        mode="system1",
        llm_api_key="present",
        embed_api_key="",
    )
    with pytest.raises(MissingCredentialsError, match="embed_api_key"):
        MemoryClient(settings=settings, llm=FakeLLMClient(responses={}))


def test_legacy_emb_mode_rejected(tmp_storage, fake_embed):
    """Passing the removed mode='emb' is rejected by Settings validation."""
    with pytest.raises(ValueError, match="removed"):
        Settings(storage_dir=tmp_storage, mode="emb")


def test_legacy_pro_alias_still_resolves(tmp_storage, fake_embed):
    """The 'pro' alias is deprecated but still resolves to system1."""
    client = MemoryClient(
        settings=Settings(storage_dir=tmp_storage, mode="pro"),
        embed=fake_embed,
        llm=FakeLLMClient(responses={}),
    )
    assert client.settings.mode == "system1"
