import asyncio
import inspect

import pytest

from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.sync_client import SyncMemoryClient
from tests.conftest import FakeLLMClient

# Async methods on MemoryClient that SyncMemoryClient does NOT mirror as blocking data ops:
#   acreate — async classmethod factory; SyncMemoryClient is built via its own __init__.
#   aclose  — async lifecycle teardown; SyncMemoryClient exposes close()/context manager.
_LIFECYCLE_ONLY = {"acreate", "aclose"}


def _annotation_str(ann) -> str:
    """Normalize an annotation to a comparable string.

    client.py uses real objects (str, int | None); sync_client.py uses PEP 563 string
    annotations ('str', 'int | None'). Render both to the same canonical form, collapsing
    the fully-qualified ``dual_mem.sdk_models.ChatMessage`` to the bare ``ChatMessage`` the
    string form carries.
    """
    if ann is inspect.Parameter.empty:
        return "<empty>"
    if isinstance(ann, str):
        s = ann
    else:
        # Use str() for parametrized generics / unions (list[str], int | None) which carry
        # no useful __name__; fall back to __name__ for plain classes (str -> "str").
        s = str(ann) if ("[" in str(ann) or "|" in str(ann)) else getattr(ann, "__name__", str(ann))
    return s.replace("dual_mem.sdk_models.", "").replace(" ", "")


def _param_fingerprint(sig: inspect.Signature) -> list[tuple]:
    """(name, kind, has_default, default_repr, annotation_str) per param, skipping self."""
    out = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        out.append(
            (
                name,
                p.kind,
                p.default is not inspect.Parameter.empty,
                repr(p.default) if p.default is not inspect.Parameter.empty else None,
                _annotation_str(p.annotation),
            )
        )
    return out


def _public_async_ops(cls) -> set[str]:
    """Public (non-underscore) async coroutine methods of cls, excluding lifecycle ones."""
    return {
        name
        for name, member in inspect.getmembers(cls, inspect.iscoroutinefunction)
        if not name.startswith("_") and name not in _LIFECYCLE_ONLY
    }


def test_sync_client_mirrors_every_async_data_op():
    """Guard against drift: every public async data op on MemoryClient must have a
    same-named sync method on SyncMemoryClient with an identical signature (params,
    defaults, kw-only-ness, annotations) — only the async-ness differs.

    This lets the sync facade keep explicit, IDE-completable methods while still failing
    CI the moment someone adds/changes a MemoryClient method without updating the wrapper.
    """
    async_ops = _public_async_ops(MemoryClient)
    sync_methods = {
        name
        for name, _ in inspect.getmembers(SyncMemoryClient, inspect.isfunction)
        if not name.startswith("_")
    }

    missing = async_ops - sync_methods
    assert not missing, (
        f"SyncMemoryClient is missing wrappers for MemoryClient methods: {sorted(missing)}. "
        f"Add an explicit blocking method that forwards via self._run(...)."
    )

    for name in sorted(async_ops):
        async_sig = inspect.signature(getattr(MemoryClient, name))
        sync_sig = inspect.signature(getattr(SyncMemoryClient, name))
        # Compare param name/kind/default/annotation as normalized strings: sync_client.py
        # uses `from __future__ import annotations` (PEP 563) so its annotations are strings
        # like 'str' while client.py's are objects like str — semantically identical.
        assert _param_fingerprint(async_sig) == _param_fingerprint(sync_sig), (
            f"Signature drift on '{name}': MemoryClient{async_sig} vs "
            f"SyncMemoryClient{sync_sig}. Keep the wrapper's params in sync."
        )


def test_sync_client_add_search(tmp_storage, fake_embed, fake_llm):
    llm = FakeLLMClient(
        responses={
            "extract": '{"identity":[],"facts":[{"content":"likes tea","confidence":0.9}],"is_ephemeral":false}',
        }
    )
    with SyncMemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=llm,
    ) as client:
        write = client.add(
            content="I prefer tea in the morning.",
            app_id="app1",
            user_id="u1",
        )
        assert write.success
        res = client.search(query="drink preference", app_ids=["app1"], user_id="u1")
        assert res.success


def test_sync_client_context_manager_closes(tmp_storage, fake_embed, fake_llm):
    client = SyncMemoryClient(
        settings=Settings(mode="system1", storage_dir=tmp_storage),
        embed=fake_embed,
        llm=fake_llm,
    )
    client.close()
    client.close()


def test_sync_client_rejects_running_loop(tmp_storage, fake_embed, fake_llm):
    async def _inner():
        client = SyncMemoryClient(
            settings=Settings(mode="system1", storage_dir=tmp_storage),
            embed=fake_embed,
            llm=fake_llm,
        )
        try:
            with pytest.raises(RuntimeError, match="event loop"):
                client.add(content="x", app_id="a", user_id="u")
        finally:
            client.close()

    asyncio.run(_inner())
