import hashlib
import math
import os

os.environ.setdefault("DUAL_MEM_LLM_API_KEY", "sk-test-fake")
os.environ.setdefault("DUAL_MEM_EMBED_API_KEY", "sk-test-fake")
os.environ.setdefault("DUAL_MEM_AUTH_DISABLED", "true")
os.environ.setdefault("DUAL_MEM_CONFIG_FILE", "/nonexistent/dual_mem_test_config.yaml")

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
    - ``chat_json`` with system containing "记忆价值评估"/"memory value gate" → key ``"gate"``
      (default passes threshold with high novelty/relevance).
    - ``chat_json`` with system containing "记忆分析专家"/"memory analyst" → key ``"extract"``
      (default ``{"facts": [], "identity": [], "intentions": [], "is_ephemeral": False}``).
    - ``chat_json`` with system containing "搜索查询生成器"/"search query generator" → key ``"search_query"``.
    - ``chat_json`` with system containing "记忆管理系统"/"memory management system" → key ``"reconcile"``.
    - Other ``chat_json`` → key ``"json"``.
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
        self.calls.append({"type": "chat_json", "system": system, "user": user, "kw": kw})
        if "记忆价值评估" in system or "memory value gate" in system:
            return self._resolve(
                "gate",
                {
                    "novelty": 0.8,
                    "biographical_relevance": 0.8,
                    "emotional_arousal": 0.3,
                    "reason": "test gate pass",
                },
                system=system,
                user=user,
            )
        if "记忆分析专家" in system or "memory analyst" in system:
            default_extract = {
                "facts": [],
                "identity": [],
                "intentions": [],
                "is_ephemeral": False,
                "gate_decision": {
                    "novelty": 0.8,
                    "biographical_relevance": 0.8,
                    "emotional_arousal": 0.3,
                    "reason": "test gate pass",
                },
            }
            return self._resolve(
                "extract",
                default_extract,
                system=system,
                user=user,
            )
        if "搜索查询生成器" in system or "search query generator" in system:
            return self._resolve("search_query", [], system=system, user=user)
        if "记忆管理系统" in system or "memory management system" in system:
            return self._resolve("reconcile", [], system=system, user=user)
        return self._resolve("json", {"facts": [], "identity": []}, system=system, user=user)

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
        self.calls.append({"type": "chat_text", "system": system, "user": user, "kw": kw})
        return self._resolve("text", "", system=system, user=user)

    async def chat_text_for_content(self, *, content: str, build_system, merge_text=None, **kw) -> str:
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
        """Scripted ReAct turn for the System2 agent.

        Routes by responses["tools"]: a callable taking ``messages`` and returning
        ``{"content": str, "tool_calls": [...]}`` or a single dict / list-of-dicts (the
        list is consumed one element per call). Default returns no tool_calls so the loop
        terminates after one turn.
        """
        self.calls.append(
            {"type": "chat_with_tools", "messages": list(messages), "tools_count": len(tools)}
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
