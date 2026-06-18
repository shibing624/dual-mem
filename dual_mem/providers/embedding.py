# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: OpenAI-compatible embedding service wrapper for single and batch text
embedding with a configurable model and output dimension.
"""
from openai import OpenAI


class EmbedService:
    """Synchronous embedding client over an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dim: int = 1536,
        timeout: int = 30,
    ):
        self.model = model
        self.dim = dim
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one vector per input (empty list short-circuits)."""
        if not texts:
            return []
        resp = self.client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dim
        )
        return [item.embedding for item in resp.data]

    def embed(self, text: str) -> list[float]:
        """Embed a single text and return its vector."""
        return self.embed_batch([text])[0]
