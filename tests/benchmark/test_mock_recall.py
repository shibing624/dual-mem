"""Mock-only recall@k harness for the hybrid reader.

Dataset shape: each case is (write_facts, query, expected_substr_in_recall). The harness
asserts recall@5 ≥ 0.6 across the small subset, which keeps regression cost low while still
catching gross retrieval breakages. Live-LLM benchmarking is out of scope for this in-repo
harness.
"""
import pytest

from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.types import Layer, MemoryNode, MemoryStatus

from conftest import FakeLLMClient


pytestmark = pytest.mark.benchmark


# Tiny S1 (time-aware) + D1 (attribute) hybrid recall cases.
_CASES = [
    # (seed_facts, query, must-contain-substring-in-recall)
    (
        [
            ("用户喜欢喝咖啡", Layer.L4_IDENTITY),
            ("用户最近爱上了滑雪", Layer.L4_IDENTITY),
            ("用户讨厌喝牛奶", Layer.L4_IDENTITY),
        ],
        "咖啡",
        "咖啡",
    ),
    (
        [
            ("用户在腾讯工作了三年", Layer.L4_IDENTITY),
            ("用户毕业于清华大学", Layer.L4_IDENTITY),
        ],
        "腾讯",
        "腾讯",
    ),
    (
        [
            ("用户最近在准备一场技术分享", Layer.L2_FACT),
            ("用户上周参加了一个 meetup", Layer.L2_FACT),
        ],
        "技术分享",
        "技术分享",
    ),
    (
        [
            ("用户家在北京", Layer.L4_IDENTITY),
            ("用户出生于1990年", Layer.L4_IDENTITY),
            ("用户喜欢旅行和摄影", Layer.L4_IDENTITY),
        ],
        "用户的家在哪里",
        "北京",
    ),
]


def _seed(client: MemoryClient, fake_embed, facts: list[tuple[str, Layer]]) -> None:
    nodes = []
    for content, layer in facts:
        n = MemoryNode(
            content=content, layer=layer,
            app_id="app", user_id="u_bench", status=MemoryStatus.ACTIVE,
        )
        n.embedding = fake_embed.embed_sync(content)
        nodes.append(n)
    client.factory.vector.upsert(nodes)


async def test_sai_mock_recall_at_k(tmp_storage, fake_embed):
    """Mock recall@5 across the SAI subset must be ≥ 0.5."""
    settings = Settings(mode="system1", storage_dir=tmp_storage)
    client = MemoryClient(
        settings=settings, embed=fake_embed,
        llm=FakeLLMClient(responses={"extract": {
            "is_ephemeral": False,
            "emotion": {"valence": 0.0, "arousal": 0.0, "dominant_emotion": None},
            "identity": [], "facts": [], "intentions": [], "basic_info": {},
        }}),
    )

    hits = 0
    for facts, query, must_contain in _CASES:
        # Fresh storage per case to avoid cross-pollination.
        client.factory.cache.bump_access([])  # noop, just sanity
        _seed(client, fake_embed, facts)
        result = await client.search(
            query=query, app_ids=["app"], user_id="u_bench", limit=5, min_score=0.0,
        )
        all_contents = [
            *(m.content for m in result.memories.profile),
            *(m.content for m in result.memories.normal),
        ]
        if any(must_contain in c for c in all_contents):
            hits += 1
        # Clean up between cases to avoid leakage.
        client.factory.vector.delete(
            [n.node_id for n in client.factory.vector.get_many(
                {"$and": [{"app_id": "app"}, {"user_id": "u_bench"}]}, limit=100)]
        )

    recall = hits / len(_CASES)
    assert recall >= 0.5, f"SAI recall@5 = {recall:.2f} < 0.5 (case hits: {hits}/{len(_CASES)})"

    await client.aclose()
