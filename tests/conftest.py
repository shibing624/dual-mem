import hashlib
import math
import os

os.environ.setdefault("DUAL_MEM_LLM_API_KEY", "sk-test-fake")
os.environ.setdefault("DUAL_MEM_EMBED_API_KEY", "sk-test-fake")
os.environ.setdefault("DUAL_MEM_AUTH_DISABLED", "true")
os.environ.setdefault("DUAL_MEM_CONFIG_FILE", "/nonexistent/dual_mem_test_config.yaml")
# Fake hash embeddings have no semantic similarity; route every candidate to Jev.
os.environ.setdefault("DUAL_MEM_RECONCILE_RELATION_THRESHOLD", "-1")

import pytest


class FakeEmbedService:
    """Deterministic embedding: same text -> same normalized vector. Async API + sync helper."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        return self.embed_sync(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_sync(t) for t in texts]

    async def embed_queued(self, text: str) -> list[float]:
        return self.embed_sync(text)

    def embed_sync(self, text: str) -> list[float]:
        """Synchronous helper used by test fixtures that need a vector without an event loop."""
        vec: list[float] = []
        counter = 0
        while len(vec) < self.dim:
            digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
            for byte in digest:
                vec.append(byte / 255.0)
                if len(vec) >= self.dim:
                    break
            counter += 1
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class FakeLLMClient:
    """Async scripted fake LLM.

    Routes by call type + system prompt keyword to entries in ``responses``:
    - ``chat_json`` whose system is an extract prompt
      (``extract_system1``: "Generate faithful explicit atomic candidates";
      ``Extractor.extract``: "只抽取来源中明确陈述" / "Extract explicit atomic statements")
      → key ``"extract"``.
    - Other ``chat_json`` (System2 derived memories) → key ``"json"``.
    - ``chat_text`` (summary) → key ``"text"`` (default ``""``).

    Each response value can be a direct result or a ``callable(system=, user=)``.
    """

    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []

    def _resolve(self, key, default, *, system, user):
        value = self.responses.get(key, default)
        if callable(value):
            return value(system=system, user=user)
        return value

    async def chat_json(self, *, system: str, user: str, **kw):
        self.calls.append(
            {"type": "chat_json", "system": system, "user": user, "kw": kw}
        )
        if any(
            marker in system
            for marker in (
                "Generate faithful explicit atomic candidates",
                "只抽取来源中明确陈述",
                "Extract explicit atomic statements",
            )
        ):
            if "extract" not in self.responses:
                return {
                    "facts": [{"content": user, "tags": []}] if user.strip() else [],
                    "identity": [],
                    "intentions": [],
                    "is_ephemeral": False,
                    "emotion": {
                        "valence": 0.0,
                        "arousal": 0.0,
                        "dominant_emotion": None,
                    },
                    "basic_info": {},
                }
            default_extract = {
                "facts": [],
                "identity": [],
                "intentions": [],
                "is_ephemeral": False,
                "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
                "basic_info": {},
            }
            return self._resolve(
                "extract",
                default_extract,
                system=system,
                user=user,
            )
        return self._resolve(
            "json", {"facts": [], "identity": []}, system=system, user=user
        )

    async def chat_json_for_content(
        self,
        *,
        content: str,
        build_system,
        merge_results,
        user: str | None = None,
        **kw,
    ):
        user_text = content if user is None else user
        system = build_system(user_text)
        return await self.chat_json(system=system, user=user_text, **kw)

    async def chat_text(self, *, system: str, user: str, **kw) -> str:
        self.calls.append(
            {"type": "chat_text", "system": system, "user": user, "kw": kw}
        )
        return self._resolve("text", "", system=system, user=user)

    async def chat_text_for_content(
        self, *, content: str, build_system, merge_text=None, **kw
    ) -> str:
        system = build_system(content)
        return await self.chat_text(system=system, user=content, **kw)

    async def chat_with_tools(
        self,
        *,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float = 0.2,
    ) -> dict:
        """Unused by the current System2 single-shot path. Kept for scripted tool-call tests.

        Routes by responses["tools"]: a callable taking ``messages`` and returning
        ``{"content": str, "tool_calls": [...]}`` or a single dict / list-of-dicts (the
        list is consumed one element per call). Default returns no tool_calls so the loop
        terminates after one turn.
        """
        self.calls.append(
            {
                "type": "chat_with_tools",
                "messages": list(messages),
                "tools_count": len(tools),
            }
        )
        spec = self.responses.get("tools")
        if spec is None:
            return {"content": "", "tool_calls": []}
        if callable(spec):
            return spec(messages=messages, tools=tools)
        if isinstance(spec, list):
            idx = sum(1 for c in self.calls if c["type"] == "chat_with_tools") - 1
            return spec[idx] if idx < len(spec) else {"content": "", "tool_calls": []}
        if isinstance(spec, dict):
            return spec
        return {"content": "", "tool_calls": []}


@pytest.fixture
def fake_embed():
    return FakeEmbedService(dim=64)


@pytest.fixture
def fake_llm():
    return FakeLLMClient()


@pytest.fixture
def tmp_storage(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    return str(storage)


class FakeJev:
    """Scripted decisions; semantic examples supply explicit outcomes, never network."""

    def __init__(self, responses=None):
        self.responses = responses
        self.calls = []
        self.max_question_chars = 60_000
        self.max_request_chars = 100_000

    async def evaluate(self, *, state, questions):
        from dual_mem.providers.jev import JevAnswer

        self.calls.append({"state": state, "questions": questions})
        response = (
            self.responses(state=state, questions=questions)
            if callable(self.responses)
            else self.responses
        )
        if isinstance(response, Exception):
            raise response
        result = {}
        for name, q in questions.items():
            choices = q.get("criteria", {})
            selection = (response or {}).get(name)
            if selection is None:
                selection = next(
                    (
                        option
                        for option in (
                            "ADD_SUPPLEMENT",
                            "NONE",
                            "RAW_ONLY",
                            "VALID",
                            "INCLUDE",
                            "FACT",
                            "CHAT",
                        )
                        if option in choices
                    ),
                    next(iter(choices), None),
                )
            if isinstance(selection, JevAnswer):
                result[name] = selection
            else:
                result[name] = JevAnswer(
                    type="choice",
                    choice=selection,
                    confidence=1,
                    probabilities={selection: 1},
                )
        return result


@pytest.fixture
def fake_jev():
    return FakeJev()


@pytest.fixture(autouse=True)
def release_chroma_systems():
    """Drop chromadb's process-global System cache after every test.

    ``chromadb.PersistentClient`` keeps one Rust System per path in
    ``SharedSystemClient._identifier_to_system`` for the whole process. Each test
    uses a fresh tmp path and many tests never ``aclose()`` their client, so the
    full suite accumulated ~hundreds of live Systems and eventually blocked inside
    native ``Bindings(...)`` around 48% (single files always passed).
    """
    yield
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except ImportError:  # pragma: no cover
        return
    for system in list(SharedSystemClient._identifier_to_system.values()):
        try:
            system.stop()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
    SharedSystemClient.clear_system_cache()


@pytest.fixture(autouse=True)
def mock_decisions_for_sdk_fixtures(monkeypatch):
    """All SDK fixtures are offline; new protocol tests inject their own script."""
    from dual_mem.registry import ComponentFactory

    original = ComponentFactory.__init__

    def initialize(self, **kwargs):
        if kwargs.get("jev") is None:
            kwargs["jev"] = FakeJev()
        original(self, **kwargs)

    monkeypatch.setattr(ComponentFactory, "__init__", initialize)
    monkeypatch.setenv("DUAL_MEM_JEV_API_KEY", "test-jev-key")
