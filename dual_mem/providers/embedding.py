# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Async OpenAI-compatible embedding service with both a direct embed/embed_batch
path (search-side, low-latency) and a write-side embed_queued path that coalesces concurrent
single requests into one batch within a 200ms window or 32-item threshold (whichever first).
"""
import asyncio
import hashlib
import logging
import time

from openai import AsyncOpenAI

from dual_mem.config import (
    CHARS_PER_TOKEN,
    EMBED_MAX_TOKENS,
    EMBED_RETRY_ATTEMPTS,
    EMBED_RETRY_BASE_DELAY,
)
from dual_mem.providers.usage import UsageCallback, UsageEvent

logger = logging.getLogger("dual_mem.embed")


def embedding_api_dimensions(model: str, dim: int) -> int | None:
    """Return ``dimensions`` for OpenAI embeddings API, or None to omit the parameter.

    qwen3-embedding / bge-m3 (vLLM served as ``embed``) reject matryoshka ``dimensions``.
    """
    name = (model or "").lower()
    if "qwen3-embedding" in name or "-for-online" in name:
        return None
    if name in ("embed",) or "bge-m3" in name or "bge_m3" in name:
        return None
    return dim if dim > 0 else None


class EmbedService:
    """Async embedding client with optional write-path batching queue."""

    DEFAULT_BATCH_SIZE = 32
    DEFAULT_BATCH_WINDOW_MS = 200.0
    DEFAULT_CACHE_SIZE = 10000

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dim: int = 1536,
        timeout: int = 30,
        queue_batch_size: int = DEFAULT_BATCH_SIZE,
        queue_batch_window_ms: float = DEFAULT_BATCH_WINDOW_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
        input_max_tokens: int = EMBED_MAX_TOKENS,
        chars_per_token: float = CHARS_PER_TOKEN,
        retry_attempts: int = EMBED_RETRY_ATTEMPTS,
        retry_base_delay: float = EMBED_RETRY_BASE_DELAY,
        usage_callback: UsageCallback | None = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self.input_max_tokens = input_max_tokens
        self.chars_per_token = chars_per_token
        self.input_max_chars = int(input_max_tokens * chars_per_token)
        self.retry_attempts = retry_attempts
        self.retry_base_delay = retry_base_delay
        self.usage_callback = usage_callback
        self._api_dimensions = embedding_api_dimensions(model, dim)
        if self._api_dimensions is None:
            logger.info(
                "EmbedService model=%r: omitting API dimensions param (native model dims); "
                "set embed_dim to the model's output size for vector store alignment.",
                model,
            )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size

        self._queue_batch_size = queue_batch_size
        self._queue_batch_window_s = queue_batch_window_ms / 1000.0
        self._queue_lock: asyncio.Lock | None = None
        self._queue_pending: list[tuple[str, asyncio.Future]] = []
        self._queue_flush_task: asyncio.Task | None = None

    @staticmethod
    def _cache_key(text: str) -> str:
        """Stable cache key for a text (md5 over utf-8 bytes)."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _maybe_cache(self, text: str, vector: list[float]) -> None:
        """Insert (text, vector) into the in-memory cache when the cache has room."""
        if len(self._cache) >= self._cache_size:
            return
        self._cache[self._cache_key(text)] = vector

    async def _embed_prepared_batch(self, prepared: list[str]) -> list[list[float]]:
        """Call the embeddings API for *prepared* strings (already chunked)."""
        kwargs: dict = {"model": self.model, "input": prepared}
        if self._api_dimensions is not None:
            kwargs["dimensions"] = self._api_dimensions

        last_exc: Exception | None = None
        start = time.perf_counter()
        for attempt in range(self.retry_attempts):
            try:
                resp = await self.client.embeddings.create(**kwargs)
                vectors = [item.embedding for item in resp.data]
                if self.usage_callback is not None:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self.usage_callback(
                        UsageEvent(
                            kind="embed_batch",
                            model=self.model,
                            latency_ms=elapsed_ms,
                            text_chars=sum(len(p) for p in prepared),
                            batch_size=len(prepared),
                        )
                    )
                return vectors
            except Exception as exc:
                last_exc = exc
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "embed_batch attempt %d/%d failed (%s: %s); retry in %.1fs",
                        attempt + 1,
                        self.retry_attempts,
                        type(exc).__name__,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return one vector per input; empty list short-circuits.

        Texts longer than ``embed_max_tokens`` (via char budget) are head-truncated to the
        budget before the API call. Over-budget inputs are L1_RAW raw-dialogue anchors that
        turn SHADOW the moment the extractor emits L2/L4 children (and SHADOW nodes are out of
        the recall pool), so the dropped tail never affects retrieval; the full text is still
        stored as the L1 document and read by the extractor.
        """
        if not texts:
            return []
        max_chars = self.input_max_chars
        prepared: list[str] = []
        n_truncated = 0
        for t in texts:
            if max_chars > 0 and len(t) > max_chars:
                prepared.append(t[:max_chars])
                n_truncated += 1
            else:
                prepared.append(t)
        if n_truncated:
            logger.info(
                "embed_batch: %d of %d inputs over limit, head-truncated to ~%d chars "
                "(≈%d tokens); raw L1 anchors only, full text kept for extraction",
                n_truncated,
                len(texts),
                max_chars,
                self.input_max_tokens,
            )
        return await self._embed_prepared_batch(prepared)

    async def embed(self, text: str) -> list[float]:
        """Embed a single text directly (no queue), using the cache when possible."""
        cached = self._cache.get(self._cache_key(text))
        if cached is not None:
            return cached
        vectors = await self.embed_batch([text])
        vector = vectors[0]
        self._maybe_cache(text, vector)
        return vector

    async def embed_queued(self, text: str) -> list[float]:
        """Embed a single text via the write-side queue: batches concurrent calls."""
        cached = self._cache.get(self._cache_key(text))
        if cached is not None:
            return cached

        if self._queue_lock is None:
            self._queue_lock = asyncio.Lock()

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        async with self._queue_lock:
            self._queue_pending.append((text, future))
            pending_count = len(self._queue_pending)
            if pending_count >= self._queue_batch_size:
                await self._flush_locked()
            elif self._queue_flush_task is None or self._queue_flush_task.done():
                self._queue_flush_task = asyncio.ensure_future(self._delayed_flush())

        vector = await future
        self._maybe_cache(text, vector)
        return vector

    async def _delayed_flush(self) -> None:
        """Wait the configured window then flush whatever is pending in the queue."""
        await asyncio.sleep(self._queue_batch_window_s)
        if self._queue_lock is not None:
            async with self._queue_lock:
                if self._queue_pending:
                    await self._flush_locked()

    async def _flush_locked(self) -> None:
        """Issue one batch embeddings call for the pending queue (caller holds the lock)."""
        if not self._queue_pending:
            return
        batch = self._queue_pending[:]
        self._queue_pending.clear()

        texts = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        try:
            vectors = await self.embed_batch(texts)
        except Exception as exc:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(exc)
            return

        for fut, vec in zip(futures, vectors):
            if not fut.done():
                fut.set_result(vec)
