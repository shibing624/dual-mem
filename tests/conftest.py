import hashlib
import math
import os

os.environ.setdefault("DUAL_MEM_LLM_API_KEY", "sk-test-fake")
os.environ.setdefault("DUAL_MEM_EMBED_API_KEY", "sk-test-fake")
os.environ.setdefault("DUAL_MEM_AUTH_DISABLED", "true")
os.environ.setdefault("DUAL_MEM_CONFIG_FILE", "/nonexistent/dual_mem_test_config.yaml")

import pytest


class FakeEmbedService:
    """Deterministic embedding: same text -> same normalized vector."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
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

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class FakeLLMClient:
    """脚本化 fake LLM。

    按调用类型 + system prompt 关键词路由到 ``responses`` 中的不同条目：
    - ``chat_with_tools``（extract）→ key ``"extract"``（兼容旧 key ``"tools"``）。
    - ``chat_json`` 且 system 含 "搜索查询生成器"/"search query generator" → key ``"search_query"``。
    - ``chat_json`` 且 system 含 "记忆管理系统"/"memory management system" → key ``"reconcile"``。
    - 其余 ``chat_json`` → key ``"json"``（默认 ``{"facts": [], "identity": []}``，保持旧测试兼容）。
    - ``chat_text``（summary）→ key ``"text"``（默认 ``""``）。

    每个 response 值可以是直接结果，也可以是 ``callable(system=, user=)``。
    """

    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []

    def _resolve(self, key, default, *, system, user):
        value = self.responses.get(key, default)
        if callable(value):
            return value(system=system, user=user)
        return value

    def chat_json(self, *, system: str, user: str, **kw) -> dict:
        self.calls.append({"type": "chat_json", "system": system, "user": user, "kw": kw})
        if "搜索查询生成器" in system or "search query generator" in system:
            return self._resolve("search_query", [], system=system, user=user)
        if "记忆管理系统" in system or "memory management system" in system:
            return self._resolve("reconcile", [], system=system, user=user)
        return self._resolve("json", {"facts": [], "identity": []}, system=system, user=user)

    def chat_with_tools(self, *, system: str, user: str, tools, **kw) -> dict:
        self.calls.append(
            {"type": "chat_with_tools", "system": system, "user": user, "tools": tools, "kw": kw}
        )
        default = {"content": '{"facts":[],"identity":[]}', "tool_calls": []}
        key = "extract" if "extract" in self.responses else "tools"
        return self._resolve(key, default, system=system, user=user)

    def chat_text(self, *, system: str, user: str, **kw) -> str:
        self.calls.append({"type": "chat_text", "system": system, "user": user, "kw": kw})
        return self._resolve("text", "", system=system, user=user)


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
