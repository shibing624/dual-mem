import time
from datetime import datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dual_mem import __version__
from dual_mem.api.schemas import (
    AddRequest,
    AddResponse,
    DeleteBulkResponse,
    DeleteResponse,
    HealthResponse,
    InfoResponse,
    PingResponse,
    SearchRequest,
    SearchResponse,
)
from dual_mem.client import MemoryClient
from dual_mem.config import Settings


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _get_client(request: Request) -> MemoryClient:
    return request.app.state.client


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
    if settings is None:
        settings = client.settings if client is not None else Settings()
    if client is None:
        client = MemoryClient(settings=settings)

    app = FastAPI(title="dual-mem REST API", version=__version__)
    app.state.settings = settings
    app.state.client = client
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

    @app.post("/v1/memories/", response_model=AddResponse, dependencies=[Depends(_verify_bearer)])
    async def add_memory(
        body: AddRequest,
        settings: Settings = Depends(_get_settings),
        client: MemoryClient = Depends(_get_client),
    ):
        _check_whitelist(settings, [body.app_id])
        if not body.content and not body.messages:
            raise HTTPException(status_code=400, detail="content 与 messages 至少二选一")
        return await client.add(
            content=body.content,
            messages=body.messages,
            app_id=body.app_id,
            user_id=body.user_id,
            agent_id=body.agent_id,
            session_id=body.session_id,
            memory_at=body.memory_at,
        )

    @app.post(
        "/v1/memories/search",
        response_model=SearchResponse,
        dependencies=[Depends(_verify_bearer)],
    )
    async def search_memory(
        body: SearchRequest,
        settings: Settings = Depends(_get_settings),
        client: MemoryClient = Depends(_get_client),
    ):
        _check_whitelist(settings, body.app_ids)
        return await client.search(
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
        )

    @app.get("/v1/memories/", dependencies=[Depends(_verify_bearer)])
    async def list_memories(
        app_id: str,
        user_id: str,
        agent_id: str = "",
        limit: int = 100,
        settings: Settings = Depends(_get_settings),
        client: MemoryClient = Depends(_get_client),
    ):
        _check_whitelist(settings, [app_id])
        return await client.list(
            app_id=app_id, user_id=user_id, agent_id=agent_id, limit=limit
        )

    @app.get("/v1/memories/{memory_id}", dependencies=[Depends(_verify_bearer)])
    async def get_memory(
        memory_id: str,
        client: MemoryClient = Depends(_get_client),
    ):
        node = await client.get(memory_id)
        if node is None:
            raise HTTPException(status_code=404, detail=f"memory_id '{memory_id}' 不存在")
        return node

    @app.delete(
        "/v1/memories/{memory_id}",
        response_model=DeleteResponse,
        dependencies=[Depends(_verify_bearer)],
    )
    async def delete_memory(
        memory_id: str,
        client: MemoryClient = Depends(_get_client),
    ):
        result = await client.delete(memory_id)
        if result["success"] is False:
            raise HTTPException(
                status_code=result["error_code"],
                detail=f"memory_id '{memory_id}' 不存在",
            )
        return result

    @app.delete(
        "/v1/memories/",
        response_model=DeleteBulkResponse,
        dependencies=[Depends(_verify_bearer)],
    )
    async def delete_bulk(
        app_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        confirm: bool = False,
        settings: Settings = Depends(_get_settings),
        client: MemoryClient = Depends(_get_client),
    ):
        _check_whitelist(settings, [app_id])
        result = await client.delete_bulk(
            app_id=app_id, user_id=user_id, agent_id=agent_id, confirm=confirm
        )
        if result["success"] is False:
            raise HTTPException(
                status_code=result["error_code"],
                detail="批量删除必须显式传入 confirm=true",
            )
        return result

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
        }

    return app
