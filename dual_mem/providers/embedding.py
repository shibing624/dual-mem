from openai import OpenAI


class EmbedService:
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
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]
