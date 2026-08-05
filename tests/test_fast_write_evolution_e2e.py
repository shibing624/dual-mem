"""E2E: fast-write -> reconcile queue -> digest -> search returns evolution_chain."""
from conftest import FakeLLMClient

from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryStatus

_JAVA = "用户最喜欢的编程语言是 Java"
_PYTHON = "用户现在主要使用 Python"
_EMOTION = {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None}


class _EvolutionLLM(FakeLLMClient):
    """Scripted extract + dynamic reconcile for Java→Python preference change."""

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
            heads = [
                n
                for n in nodes
                if n.layer == Layer.L4_IDENTITY
                and n.is_latest
                and n.status in (MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED)
            ]
            java_heads = [
                n
                for n in heads
                if _JAVA in n.content and n.is_latest and n.status == MemoryStatus.ACTIVE
            ]
            if java_heads:
                return [
                    {
                        "reason": "语言偏好从 Java 转向 Python",
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
                    "reason": "首次记录语言偏好",
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


async def test_fast_write_reconcile_digest_evolution_chain(tmp_storage, fake_embed):
    llm = _EvolutionLLM()
    client = MemoryClient(
        settings=Settings(
            mode="dual",
            storage_dir=tmp_storage,
            reconcile_sync=False,
        ),
        embed=fake_embed,
        llm=llm,
    )
    llm.factory = client.factory

    await client.add(content="第一轮：Java 偏好", app_id="app", user_id="u")
    digest1 = await client.digest()
    assert digest1.processed >= 1

    await client.add(content="第二轮：转向 Python", app_id="app", user_id="u")
    digest2 = await client.digest()
    assert digest2.processed >= 1

    res = await client.search(
        query="用户现在用什么编程语言？",
        app_ids=["app"],
        user_id="u",
        min_score=0.0,
        profile_limit=5,
    )
    all_items = res.memories.profile + res.memories.normal
    evolved = [m for m in all_items if m.evolution_chain]
    assert evolved, "expected at least one hit with evolution_chain after reconcile digest"
    head = evolved[0]
    assert _PYTHON in head.content
    chain_contents = [node.content for node in head.evolution_chain]
    assert any(_JAVA in c for c in chain_contents)
