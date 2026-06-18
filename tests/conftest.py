import hashlib
import math
import os

os.environ.setdefault("DUAL_MEM_LLM_API_KEY", "sk-test-fake")
os.environ.setdefault("DUAL_MEM_EMBED_API_KEY", "sk-test-fake")
os.environ.setdefault("DUAL_MEM_AUTH_DISABLED", "true")

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
    def __init__(self, responses: dict | None = None):
        self.responses = responses or {}
        self.calls: list[dict] = []

    def chat_json(self, *, system: str, user: str, **kw) -> dict:
        self.calls.append({"type": "chat_json", "system": system, "user": user, "kw": kw})
        return self.responses.get("json", {"facts": [], "identity": []})

    def chat_with_tools(self, *, system: str, user: str, tools, **kw) -> dict:
        self.calls.append(
            {"type": "chat_with_tools", "system": system, "user": user, "tools": tools, "kw": kw}
        )
        return self.responses.get(
            "tools", {"content": '{"facts":[],"identity":[]}', "tool_calls": []}
        )


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
