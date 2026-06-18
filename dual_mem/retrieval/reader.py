from dual_mem.isolation import build_filter
from dual_mem.registry import ComponentFactory
from dual_mem.types import Layer, MemoryNode, MemoryStatus

NORMAL_LAYERS = [Layer.L1_RAW, Layer.L2_FACT, Layer.L5_KNOWLEDGE, Layer.L3_SUMMARY]


class Reader:
    def __init__(self, *, factory: ComponentFactory):
        self.factory = factory

    async def search(
        self,
        *,
        query: str,
        app_ids: list[str],
        user_id: str,
        agent_ids: list[str] | None = None,
        session_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.4,
        profile_limit: int = -1,
        profile_min_score: float = 0.3,
    ) -> dict:
        embedding = self.factory.embed.embed(query)
        where = build_filter(
            app_ids=app_ids,
            user_id=user_id,
            agent_ids=agent_ids,
            session_ids=session_ids,
            layers=NORMAL_LAYERS,
            statuses=[MemoryStatus.ACTIVE],
        )
        hits = self.factory.vector.query(embedding=embedding, where=where, top_k=limit)
        normal = [
            self._to_dict(node) for node in hits if node.score >= min_score
        ][:limit]
        return {"profile": [], "proactive": [], "normal": normal}

    @staticmethod
    def _to_dict(node: MemoryNode) -> dict:
        return {
            "memory_id": node.node_id,
            "content": node.content,
            "category": node.category.value,
            "score": round(node.score, 4),
            "tags": node.tags,
            "memory_at": node.memory_at,
            "gmt_created": node.gmt_created,
            "gmt_modified": node.gmt_modified,
        }
