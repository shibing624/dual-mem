"""Optional cross-encoder provider using the Cohere/Jina-compatible rerank contract."""
from __future__ import annotations

import math

import httpx


class RerankClient:
    def __init__(self, *, endpoint: str, model: str = "", api_key: str = "",
                 timeout: float = 30, http_client: httpx.AsyncClient | None = None):
        self.endpoint = endpoint
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def rerank(self, *, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        body = {"query": query, "documents": documents, "top_n": len(documents),
                "return_documents": False}
        if self.model:
            body["model"] = self.model
        response = await self._client.post(self.endpoint, headers=self.headers, json=body)
        response.raise_for_status()
        rows = response.json().get("results")
        if not isinstance(rows, list) or len(rows) != len(documents):
            raise ValueError("reranker must return exactly one score per document")
        scores: dict[int, float] = {}
        for row in rows:
            index = row.get("index")
            score = row.get("relevance_score")
            if (not isinstance(index, int) or isinstance(index, bool)
                    or not 0 <= index < len(documents) or index in scores
                    or not isinstance(score, (int, float)) or isinstance(score, bool)
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("invalid rerank index or relevance score")
            scores[index] = float(score)
        return [scores[i] for i in range(len(documents))]

    async def aclose(self):
        if self._owns_client:
            await self._client.aclose()
