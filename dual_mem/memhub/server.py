# -*- coding: utf-8 -*-
"""Loopback-first MemHub Core: bearer HTTP, durable evidence and one writer."""

from __future__ import annotations

import fcntl
import os
import secrets
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dual_mem import __version__
from dual_mem.client import MemoryClient
from dual_mem.config import Settings
from dual_mem.errors import IdempotencyConflictError
from dual_mem.memhub.config import (
    hub_root,
    hub_settings,
    load_or_create_runtime_token,
    publish_runtime,
    unpublish_runtime,
)
from dual_mem.memhub.queue import HubWriteQueue
from dual_mem.sdk_models import SearchMemories
from dual_mem.types import Layer, MemoryStatus
from dual_mem.isolation import build_filter

HUB_USER_ID = "default"


class HubWriteRequest(BaseModel):
    content: str = ""
    messages: list[dict] | None = None
    session_id: str = ""
    memory_at: int | None = None
    request_id: str
    raw_only: bool = False
    # ``False`` returns as soon as the evidence is durable; the job runs later.
    wait: bool = False


class HubRecallRequest(BaseModel):
    query: str
    limit: int = 10
    raw_limit: int = 5


class HubProcessLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a")
        try:
            fcntl.flock(
                self._handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                f"another MemHub Core already owns {self._path.parent}"
            ) from exc

    def release(self) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def _instance_id(root: Path) -> str:
    path = root / "instance-id"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    value = str(uuid.uuid4())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value + "\n")
    return value


def create_hub_app(
    *,
    api_key: str | None = None,
    hub_dir: str | Path | None = None,
    client: MemoryClient | None = None,
    settings: Settings | None = None,
    published_base_url: str = "http://127.0.0.1:8765",
    queue_retry_seconds: float = 30.0,
) -> FastAPI:
    """Create the personal Hub Core; one process owns one local store.

    ``api_key`` (else the persisted runtime token, created on first start) is
    written to the token file inside the lifespan, after the exclusive lock, so a
    second ``hub serve`` cannot replace the running Core's token on its way out.
    """
    root = hub_root(hub_dir)
    lock = HubProcessLock(root / "hub.lock")
    owns_client = client is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        lock.acquire()
        core_client: MemoryClient | None = None
        queue: HubWriteQueue | None = None
        try:
            resolved_key = load_or_create_runtime_token(root, token=api_key)
            app.state.api_key = resolved_key
            publish_runtime(
                base_url=published_base_url,
                instance_id=app.state.instance_id,
                pid=os.getpid(),
                hub_dir=root,
            )
            core_client = client or MemoryClient(
                settings=settings or hub_settings(root)
            )
            app.state.client = core_client

            async def process(payload: dict[str, Any]) -> dict[str, Any]:
                kwargs = {
                    "content": payload["content"],
                    "messages": payload["messages"],
                    "user_id": HUB_USER_ID,
                    "agent_id": "",
                    "session_id": payload["session_id"],
                    "memory_at": payload["memory_at"],
                    "request_id": payload["request_id"],
                }
                if payload["raw_only"]:
                    result = await core_client.add_raw(**kwargs)
                else:
                    result = await core_client.add(**kwargs)
                return result.to_dict()

            queue = HubWriteQueue(
                root / "hub.db", process, retry_base_seconds=queue_retry_seconds
            )
            app.state.queue = queue
            queue.start()
            yield
        finally:
            if queue is not None:
                await queue.close()
            if owns_client and core_client is not None:
                await core_client.aclose()
            unpublish_runtime(app.state.instance_id, os.getpid(), root)
            lock.release()

    app = FastAPI(
        title="dual-mem MemHub Core",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.instance_id = _instance_id(root)
    app.state.api_key = api_key or ""

    def verify(
        authorization: str | None = Header(default=None),
    ) -> None:
        expected = app.state.api_key
        if not expected:
            raise HTTPException(status_code=503, detail="Hub Core is starting")
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise HTTPException(status_code=401, detail="missing Hub bearer token")
        supplied = authorization[len(prefix) :].strip()
        if not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=403, detail="invalid Hub bearer token")

    @app.exception_handler(IdempotencyConflictError)
    async def idempotency_conflict(_request, exc: IdempotencyConflictError):
        return JSONResponse(
            status_code=409,
            content={"success": False, "error": str(exc)},
        )

    @app.get("/health")
    async def health():
        return {"status": "SERVING"}

    @app.get("/v1/hub/identity", dependencies=[Depends(verify)])
    async def identity():
        resolved = settings or (client.settings if client else hub_settings(root))
        return {
            "instance_id": app.state.instance_id,
            "sdk_version": __version__,
            "scope": {
                "app_id": resolved.default_app_id,
                "user_id": HUB_USER_ID,
                "agent_id": "",
            },
        }

    @app.post("/v1/hub/writes", dependencies=[Depends(verify)])
    async def submit_write(body: HubWriteRequest):
        if not body.content.strip() and not body.messages:
            raise HTTPException(
                status_code=400,
                detail="content and messages cannot both be empty",
            )
        job = app.state.queue.submit(body.model_dump(exclude={"wait"}))
        if body.wait:
            job = await app.state.queue.wait(body.request_id)
        return job

    @app.get("/v1/hub/writes/{request_id}", dependencies=[Depends(verify)])
    async def write_status(request_id: str):
        try:
            return app.state.queue.get(request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Hub write not found") from exc

    @app.post("/v1/hub/extract", dependencies=[Depends(verify)])
    async def extract_raw():
        """Queue extraction for evidence stored with ``raw_only``."""
        return {"queued": app.state.queue.extract_raw()}

    @app.post("/v1/hub/retry", dependencies=[Depends(verify)])
    async def retry_failed():
        return {"queued": app.state.queue.retry_failed()}

    @app.post("/v1/hub/recall", dependencies=[Depends(verify)])
    async def recall(body: HubRecallRequest):
        core_client: MemoryClient = app.state.client
        memories = SearchMemories()
        conversations = SearchMemories()
        if body.limit > 0:
            memories = (
                await core_client.search(
                    query=body.query,
                    user_id=HUB_USER_ID,
                    agent_ids=[""],
                    limit=body.limit,
                )
            ).memories
        if body.raw_limit > 0:
            conversations = (
                await core_client.search_conversation(
                    query=body.query,
                    user_id=HUB_USER_ID,
                    agent_ids=[""],
                    limit=body.raw_limit,
                )
            ).memories
        return {
            "memories": memories.to_dict(),
            "conversations": [item.to_dict() for item in conversations.normal],
        }

    @app.get("/v1/hub/status", dependencies=[Depends(verify)])
    async def status():
        core_client: MemoryClient = app.state.client
        nodes = core_client.factory.vector.get_many(
            build_filter(
                app_ids=[core_client.settings.default_app_id],
                user_id=HUB_USER_ID,
                agent_ids=[""],
                statuses=[MemoryStatus.ACTIVE],
            ),
            limit=None,
        )
        latest = [node for node in nodes if node.is_latest]
        return {
            "instance_id": app.state.instance_id,
            "root": str(root),
            "memories": {
                "versions": len(nodes),
                "current": len(latest),
                "by_layer": dict(Counter(node.layer.value for node in latest)),
                "raw_by_source": dict(
                    Counter(
                        node.session_id.partition(":")[0]
                        for node in latest
                        if node.layer == Layer.L0_RAW
                    )
                ),
            },
            "writes": app.state.queue.counts(),
        }

    return app
