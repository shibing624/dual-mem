import pytest
from typer.testing import CliRunner

from dual_mem import MemoryClient
from dual_mem.cli import main as cli_main
from dual_mem.config import Settings

runner = CliRunner()


@pytest.fixture
def patched_client(tmp_storage, fake_embed, fake_llm, monkeypatch):
    settings = Settings(storage_dir=tmp_storage, mode="system1", auth_disabled=True)
    client = MemoryClient(settings=settings, embed=fake_embed, llm=fake_llm)
    monkeypatch.setattr(cli_main, "make_client", lambda mode=None: client)
    return client


def test_help():
    result = runner.invoke(cli_main.app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("add", "search", "list", "get", "delete", "digest", "serve", "mcp"):
        assert cmd in result.output


def test_add_then_search(patched_client):
    added = runner.invoke(
        cli_main.app,
        ["add", "--content", "用户喜欢喝咖啡", "--app-id", "app", "--user-id", "u"],
    )
    assert added.exit_code == 0
    assert "memory_id" in added.output

    searched = runner.invoke(
        cli_main.app,
        ["search", "用户喜欢喝咖啡", "--app-id", "app", "--user-id", "u"],
    )
    assert searched.exit_code == 0
    assert "用户喜欢喝咖啡" in searched.output
