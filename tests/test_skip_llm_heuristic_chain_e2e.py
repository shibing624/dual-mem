"""E2E: skip_llm fast-path + zero-LLM heuristic chain-linking.

On the ``reconcile_skip_llm`` path the reconcile LLM is never called, so preference-update
nodes normally get no ``supersedes``/``superseded_by`` pointers. ``link_evolution_chains_heuristic``
back-fills those pointers by grouping ACTIVE L2/L4 nodes by shared tag and wiring
``older <-supersedes- newer`` in ``keep_active=True`` mode: old nodes stay ACTIVE/recallable while
the timeline becomes reconstructable.
"""
from conftest import FakeLLMClient

from dual_mem import MemoryClient
from dual_mem.config import Settings
from dual_mem.system2.reconciler_worker import link_evolution_chains_heuristic
from dual_mem.types import Layer, MemoryNode, MemoryStatus

_JAVA = "用户最喜欢的编程语言是 Java"
_PYTHON = "用户现在主要使用 Python"
_EMOTION = {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None}


class _SkipLLMExtractLLM(FakeLLMClient):
    """Scripted extractor emitting two same-tag L4 identity nodes; reconcile never runs."""

    def __init__(self) -> None:
        super().__init__()
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
        # reconcile prompt must never be reached on the skip_llm path
        if "记忆管理系统" in system or "memory management system" in system:
            raise AssertionError("reconcile LLM must not run when reconcile_skip_llm=True")
        return self._resolve("json", {"facts": [], "identity": []}, system=system, user=user)


async def test_skip_llm_heuristic_chain_links_same_tag_nodes(tmp_storage, fake_embed):
    llm = _SkipLLMExtractLLM()
    client = MemoryClient(
        settings=Settings(
            mode="dual",
            storage_dir=tmp_storage,
            system2_trigger_mode="manual",
            reconcile_sync=False,
            reconcile_skip_llm=True,
            reconcile_link_chains_heuristic=True,
        ),
        embed=fake_embed,
        llm=llm,
    )

    await client.add(content="第一轮：Java 偏好", app_id="app", user_id="u")
    await client.digest()
    await client.add(content="第二轮：转向 Python", app_id="app", user_id="u")
    await client.digest()

    nodes = client.factory.vector.get_many(
        {"$and": [{"app_id": "app"}, {"user_id": "u"}]},
        limit=50,
    )
    identity = [n for n in nodes if n.layer == Layer.L4_IDENTITY]
    java = next(n for n in identity if _JAVA in n.content)
    python = next(n for n in identity if _PYTHON in n.content)

    # heuristic wired the chain: newer supersedes older
    assert java.node_id in (python.supersedes or [])
    assert python.node_id in (java.superseded_by or [])
    # keep_active mode: the old node stays ACTIVE and recallable, not hidden
    assert java.status == MemoryStatus.ACTIVE
    assert java.is_latest is True


async def test_skip_llm_heuristic_chain_surfaces_in_search(tmp_storage, fake_embed):
    llm = _SkipLLMExtractLLM()
    client = MemoryClient(
        settings=Settings(
            mode="dual",
            storage_dir=tmp_storage,
            system2_trigger_mode="manual",
            reconcile_sync=False,
            reconcile_skip_llm=True,
            reconcile_link_chains_heuristic=True,
        ),
        embed=fake_embed,
        llm=llm,
    )

    await client.add(content="第一轮：Java 偏好", app_id="app", user_id="u")
    await client.digest()
    await client.add(content="第二轮：转向 Python", app_id="app", user_id="u")
    await client.digest()

    res = await client.search(
        query="用户现在用什么编程语言？",
        app_ids=["app"],
        user_id="u",
        min_score=0.0,
        profile_limit=5,
    )
    all_items = res.memories.profile + res.memories.normal
    evolved = [m for m in all_items if m.evolution_chain]
    assert evolved, "expected a hit carrying evolution_chain after heuristic linking"
    chain_contents = [
        node.content for m in evolved for node in m.evolution_chain
    ] + [m.content for m in evolved]
    assert any(_JAVA in c for c in chain_contents)
    assert any(_PYTHON in c for c in chain_contents)


async def test_heuristic_disabled_leaves_nodes_unlinked(tmp_storage, fake_embed):
    llm = _SkipLLMExtractLLM()
    client = MemoryClient(
        settings=Settings(
            mode="dual",
            storage_dir=tmp_storage,
            system2_trigger_mode="manual",
            reconcile_sync=False,
            reconcile_skip_llm=True,
            reconcile_link_chains_heuristic=False,
        ),
        embed=fake_embed,
        llm=llm,
    )

    await client.add(content="第一轮：Java 偏好", app_id="app", user_id="u")
    await client.digest()
    await client.add(content="第二轮：转向 Python", app_id="app", user_id="u")
    await client.digest()

    nodes = client.factory.vector.get_many(
        {"$and": [{"app_id": "app"}, {"user_id": "u"}]},
        limit=50,
    )
    identity = [n for n in nodes if n.layer == Layer.L4_IDENTITY]
    # with the heuristic off, no supersedes edges are wired
    assert all(not (n.supersedes or []) for n in identity)


async def test_heuristic_does_not_link_across_layers_or_blank_tags(
    tmp_storage,
    fake_embed,
):
    client = MemoryClient(
        settings=Settings(
            mode="dual",
            storage_dir=tmp_storage,
            system2_trigger_mode="manual",
        ),
        embed=fake_embed,
        llm=FakeLLMClient(),
    )
    vector = client.factory.vector
    vector.upsert(
        [
            MemoryNode(
                node_id="fact",
                app_id="app",
                user_id="u",
                layer=Layer.L2_FACT,
                content="fact",
                embedding=[1.0, 0.0],
                tags=["lang"],
            ),
            MemoryNode(
                node_id="identity",
                app_id="app",
                user_id="u",
                layer=Layer.L4_IDENTITY,
                content="identity",
                embedding=[1.0, 0.0],
                tags=["lang"],
            ),
            MemoryNode(
                node_id="blank-old",
                app_id="app",
                user_id="u",
                layer=Layer.L4_IDENTITY,
                content="blank old",
                embedding=[1.0, 0.0],
                tags=[" "],
            ),
            MemoryNode(
                node_id="blank-new",
                app_id="app",
                user_id="u",
                layer=Layer.L4_IDENTITY,
                content="blank new",
                embedding=[1.0, 0.0],
                tags=[""],
            ),
        ]
    )

    linked = link_evolution_chains_heuristic(
        client.factory,
        app_id="app",
        user_id="u",
    )

    assert linked == 0