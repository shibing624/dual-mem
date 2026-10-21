# -*- coding: utf-8 -*-
"""Async HTTP client for the single-writer MemHub Core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from dual_mem.memhub.config import HubConfig, resolve_hub_config


class HubClient:
    """Thin bearer-authenticated client; it never opens the memory store."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 120,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Hub API key missing: start local `dual-mem hub serve` or run "
                "`dual-mem hub attach --base-url ... --api-key ...`"
            )
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    @classmethod
    def from_config(
        cls,
        hub_dir: str | Path | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> HubClient:
        cfg = resolve_hub_config(hub_dir)
        return cls(
            base_url=cfg.resolved_base_url(),
            api_key=cfg.resolved_api_key(),
            transport=transport,
        )

    async def __aenter__(self) -> HubClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    async def identity(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/hub/identity")

    async def write(
        self,
        *,
        content: str = "",
        messages: list[dict] | None = None,
        session_id: str = "",
        memory_at: int | None = None,
        request_id: str,
        raw_only: bool = False,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/hub/writes",
            json={
                "content": content,
                "messages": messages,
                "session_id": session_id,
                "memory_at": memory_at,
                "request_id": request_id,
                "raw_only": raw_only,
            },
        )

    async def write_status(self, request_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/hub/writes/{request_id}")

    async def recall(
        self,
        *,
        query: str,
        limit: int = 10,
        raw_limit: int = 5,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/hub/recall",
            json={"query": query, "limit": limit, "raw_limit": raw_limit},
        )

    async def status(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/hub/status")

    async def aclose(self) -> None:
        await self._http.aclose()


def hub_client_config(hub_dir: str | Path | None = None) -> HubConfig:
    """Public helper for callers that need the resolved attachment metadata."""
    return resolve_hub_config(hub_dir)
