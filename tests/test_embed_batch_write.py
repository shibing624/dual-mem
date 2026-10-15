from conftest import FakeEmbedService, FakeLLMClient, FakeJev
from dual_mem import MemoryClient
from dual_mem.config import Settings


class CountingEmbed(FakeEmbedService):
    def __init__(self):
        super().__init__()
        self.batches = []
        self.direct = []

    async def embed(self, text):
        self.direct.append(text)
        return await super().embed(text)

    async def embed_batch(self, texts):
        self.batches.append(texts)
        return await super().embed_batch(texts)


async def test_one_raw_embedding_and_single_structured_batch(tmp_storage):
    embed = CountingEmbed()
    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=embed,
        jev=FakeJev(),
        llm=FakeLLMClient(
            {
                "extract": {
                    "facts": [{"content": "80 yuan"}, {"content": "30 yuan"}],
                    "identity": [{"content": "Lives in Beijing"}],
                }
            }
        ),
    )
    result = await c.add(content="raw evidence", user_id="u")
    assert embed.direct == ["raw evidence"] and len(embed.batches) == 1
    assert len(embed.batches[0]) == 3 and result.extracted_count == 3


async def test_known_skip_needs_no_structured_embedding(tmp_storage):
    embed = CountingEmbed()
    jev = FakeJev()
    c = MemoryClient(
        settings=Settings(storage_dir=tmp_storage),
        embed=embed,
        jev=jev,
        llm=FakeLLMClient({"extract": {"facts": [{"content": "Known"}]}}),
    )
    original = await c.add(content="Known", user_id="u")
    jev.responses = {"p0:action": "SKIP", "p0:target": original.extra_node_ids[0]}
    result = await c.add(content="Known", user_id="u")
    assert result.extracted_count == 0 and not result.is_ephemeral
    assert len(embed.batches) == 1 and len(embed.direct) == 2
