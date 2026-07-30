"""reconcile_sync=True: evolution chain is visible immediately after write (no digest)."""
from conftest import FakeLLMClient

from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryStatus

_JAVA = "用户最喜欢的编程语言是 Java"
_PYTHON = "用户现在主要使用 Python"
_EMOTION = {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None}


class _SyncEvolutionLLM(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.factory = None
        self._extract_idx = 0

    async def chat_json(self, *, system: str, user: str, **kw):
        self.calls.append({"type": "chat_json", "system": system, "user": user, "kw": kw})
        if "记忆分析专家" in system or "memory analyst" in system:
            content = _JAVA if self._extract_idx == 0 else _PYTHON
            self._extract_idx += 1
            return {
                "is_ephemeral": False,
                "emotion": _EMOTION,
                "identity": [{"content": content, "speculate": None, "tags": ["lang"]}],
                "facts": [],
                "intentions": [],
                "basic_info": {},
            }
        if "记忆管理系统" in system or "memory management system" in system:
            assert self.factory is not None
            nodes = self.factory.vector.get_many(
                {"$and": [{"app_id": "app"}, {"user_id": "u"}]},
                limit=50,
            )
            java_heads = [
                n
                for n in nodes
                if n.layer == Layer.L4_IDENTITY
                and n.is_latest
                and n.status == MemoryStatus.ACTIVE
                and _JAVA in n.content
            ]
            if java_heads:
                return [
                    {
                        "reason": "语言偏好变化",
                        "ops": [
                            {
                                "op": "ADD",
                                "content": _PYTHON,
                                "layer": "L4_IDENTITY",
                                "supersedes": [java_heads[0].node_id],
                                "tags": ["lang"],
                            }
                        ],
                    }
                ]
            return [
                {
                    "reason": "首次记录",
                    "ops": [
                        {
                            "op": "ADD",
                            "content": _JAVA,
                            "layer": "L4_IDENTITY",
                            "supersedes": [],
                            "tags": ["lang"],
                        }
                    ],
                }
            ]
        return self._resolve("json", {"facts": [], "identity": []}, system=system, user=user)


async def test_reconcile_sync_inline_evolution_without_digest(tmp_storage, fake_embed):
    llm = _SyncEvolutionLLM()
    client = MemoryClient(
        settings=Settings(
            mode="system1",
            storage_dir=tmp_storage,
            reconcile_sync=True,
        ),
        embed=fake_embed,
        llm=llm,
    )
    llm.factory = client.factory

    await client.add(content="Java 时代", app_id="app", user_id="u")
    await client.add(content="Python 时代", app_id="app", user_id="u")

    res = await client.search(
        query="编程语言偏好",
        app_ids=["app"],
        user_id="u",
        min_score=0.0,
        profile_limit=5,
    )
    items = res.memories.profile + res.memories.normal
    evolved = [m for m in items if m.evolution_chain]
    assert evolved
    assert _PYTHON in evolved[0].content
    assert any(_JAVA in node.content for node in evolved[0].evolution_chain)
