# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: FastAPI app — HTTP front-end over MemoryOperations (same contract as MCP tools).
"""
import time
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dual_mem import __version__
from dual_mem.api.contracts import MEMORY_TOOL_CONTRACTS
from dual_mem.api.operations import MemoryOperations
from dual_mem.api.schemas import (
    AddRequest,
    AddResponse,
    CapabilitiesResponse,
    DeleteBulkResponse,
    DeleteResponse,
    DigestResponse,
    HealthResponse,
    InfoResponse,
    PingResponse,
    SearchRequest,
    SearchResponse,
    UpdateRequest,
    UpdateResponse,
)
from dual_mem.client import MemoryClient
from dual_mem.config import Settings


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _get_ops(request: Request) -> MemoryOperations:
    return request.app.state.ops


def _verify_bearer(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    settings: Settings = request.app.state.settings
    if settings.auth_disabled:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="缺少有效的 Authorization Bearer 凭证")
    token = authorization[len("Bearer ") :].strip()
    if not token:
        raise HTTPException(status_code=403, detail="缺少有效的 Authorization Bearer 凭证")


def _check_whitelist(settings: Settings, app_ids: list[str]) -> None:
    if settings.auth_disabled:
        return
    for app_id in app_ids:
        if app_id not in settings.app_whitelist:
            raise HTTPException(status_code=403, detail=f"app_id '{app_id}' 不在白名单中")


def create_app(
    *, client: MemoryClient | None = None, settings: Settings | None = None
) -> FastAPI:
    """Build FastAPI app wired to MemoryOperations (REST ≡ MCP tool contract)."""
    if settings is None:
        settings = client.settings if client is not None else Settings()
    if client is None:
        client = MemoryClient(settings=settings)

    ops = MemoryOperations(client)

    app = FastAPI(
        title="dual-mem REST API",
        version=__version__,
        description="HTTP transport for dual-mem memory tools; see GET /v1/capabilities",
    )
    app.state.settings = settings
    app.state.client = client
    app.state.ops = ops
    app.state.started_at = time.time()

    @app.exception_handler(HTTPException)
    async def _http_exc_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error_code": exc.status_code,
                "error_message": str(exc.detail),
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exc_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_code": 400,
                "error_message": str(exc.errors()),
            },
        )

    # ── memory_add ──────────────────────────────────────────────────────────

    @app.post("/v1/memories/", response_model=AddResponse, dependencies=[Depends(_verify_bearer)])
    async def memory_add(
        body: AddRequest,
        settings: Settings = Depends(_get_settings),
        ops: MemoryOperations = Depends(_get_ops),
    ):
        _check_whitelist(settings, [body.app_id])
        if not body.content and not body.messages:
            raise HTTPException(status_code=400, detail="content 与 messages 至少二选一")
        return await ops.memory_add(
            content=body.content,
            messages=body.messages,
            app_id=body.app_id,
            user_id=body.user_id,
            agent_id=body.agent_id,
            session_id=body.session_id,
            memory_at=body.memory_at,
        )

    # ── memory_search ─────────────────────────────────────────────────────────

    @app.post(
        "/v1/memories/search",
        response_model=SearchResponse,
        dependencies=[Depends(_verify_bearer)],
    )
    async def memory_search(
        body: SearchRequest,
        settings: Settings = Depends(_get_settings),
        ops: MemoryOperations = Depends(_get_ops),
    ):
        _check_whitelist(settings, body.app_ids)
        return await ops.memory_search(
            query=body.query,
            app_ids=body.app_ids,
            user_id=body.user_id,
            agent_ids=body.agent_ids,
            session_ids=body.session_ids,
            limit=body.limit,
            min_score=body.min_score,
            profile_limit=body.profile_limit,
            profile_min_score=body.profile_min_score,
            intention_limit=body.intention_limit,
            created_after=body.created_after,
            debug=body.debug,
        )

    # ── memory_list ───────────────────────────────────────────────────────────

    @app.get("/v1/memories/", dependencies=[Depends(_verify_bearer)])
    async def memory_list(
        app_id: str,
        user_id: str,
        agent_id: str = "",
        limit: int = 100,
        settings: Settings = Depends(_get_settings),
        ops: MemoryOperations = Depends(_get_ops),
    ):
        _check_whitelist(settings, [app_id])
        return await ops.memory_list(
            app_id=app_id, user_id=user_id, agent_id=agent_id, limit=limit
        )

    # ── memory_get ────────────────────────────────────────────────────────────

    @app.get("/v1/memories/{memory_id}", dependencies=[Depends(_verify_bearer)])
    async def memory_get(
        memory_id: str,
        ops: MemoryOperations = Depends(_get_ops),
    ):
        item = await ops.memory_get(memory_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"memory_id '{memory_id}' 不存在")
        return item

    # ── memory_update ─────────────────────────────────────────────────────────

    @app.put(
        "/v1/memories/{memory_id}",
        response_model=UpdateResponse,
        dependencies=[Depends(_verify_bearer)],
    )
    async def memory_update(
        memory_id: str,
        body: UpdateRequest,
        ops: MemoryOperations = Depends(_get_ops),
    ):
        result = await ops.memory_update(memory_id, body.content)
        if not result["success"]:
            raise HTTPException(
                status_code=result.get("error_code") or 404,
                detail=f"memory_id '{memory_id}' 不存在",
            )
        return result

    # ── memory_delete ─────────────────────────────────────────────────────────

    @app.delete(
        "/v1/memories/{memory_id}",
        response_model=DeleteResponse,
        dependencies=[Depends(_verify_bearer)],
    )
    async def memory_delete(
        memory_id: str,
        ops: MemoryOperations = Depends(_get_ops),
    ):
        result = await ops.memory_delete(memory_id)
        if not result["success"]:
            raise HTTPException(
                status_code=result.get("error_code") or 404,
                detail=f"memory_id '{memory_id}' 不存在",
            )
        return result

    # ── memory_delete_scope ───────────────────────────────────────────────────

    @app.delete(
        "/v1/memories/",
        response_model=DeleteBulkResponse,
        dependencies=[Depends(_verify_bearer)],
    )
    async def memory_delete_scope(
        app_id: str,
        confirm: bool = False,
        user_id: str | None = None,
        agent_id: str | None = None,
        settings: Settings = Depends(_get_settings),
        ops: MemoryOperations = Depends(_get_ops),
    ):
        _check_whitelist(settings, [app_id])
        result = await ops.memory_delete_scope(
            app_id=app_id,
            user_id=user_id,
            agent_id=agent_id,
            confirm=confirm,
        )
        if not result["success"]:
            raise HTTPException(
                status_code=result.get("error_code") or 400,
                detail="批量删除必须显式传入 confirm=true",
            )
        return result

    # ── memory_list_scopes ────────────────────────────────────────────────────

    @app.get("/v1/scopes/", dependencies=[Depends(_verify_bearer)])
    async def memory_list_scopes(
        app_id: str | None = None,
        limit: int = 5000,
        settings: Settings = Depends(_get_settings),
        ops: MemoryOperations = Depends(_get_ops),
    ):
        if app_id is not None:
            _check_whitelist(settings, [app_id])
        return await ops.memory_list_scopes(app_id=app_id, limit=limit)

    # ── memory_digest ─────────────────────────────────────────────────────────

    @app.post(
        "/v1/digest/",
        response_model=DigestResponse,
        dependencies=[Depends(_verify_bearer)],
    )
    async def memory_digest(ops: MemoryOperations = Depends(_get_ops)):
        return await ops.memory_digest()

    # ── discovery (npm / TS MCP codegen) ──────────────────────────────────────

    @app.get("/v1/capabilities", response_model=CapabilitiesResponse)
    async def capabilities():
        return {
            "sdk_version": __version__,
            "mode": app.state.settings.mode,
            "tools": MEMORY_TOOL_CONTRACTS,
            "openapi_url": "/openapi.json",
        }

    # ── ops meta ──────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return {
            "status": "SERVING",
            "message": "Service is healthy",
            "uptime_seconds": round(time.time() - app.state.started_at, 2),
        }

    @app.get("/ping", response_model=PingResponse)
    async def ping():
        now = time.time()
        return {
            "message": "pong",
            "server_time": datetime.fromtimestamp(now).isoformat(),
            "timestamp": int(now * 1000),
        }

    @app.get("/info", response_model=InfoResponse)
    async def info():
        return {
            "sdk_version": __version__,
            "mode": app.state.settings.mode,
            "build": "dual-mem",
            "capabilities_url": "/v1/capabilities",
        }

    @app.on_event("shutdown")
    async def _on_shutdown():
        await ops.aclose()

    return app
