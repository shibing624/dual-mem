# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Pydantic request/response schemas for the dual-mem REST API endpoints.
"""
from pydantic import BaseModel, Field


class AddRequest(BaseModel):
    """Request body for adding a memory (raw content or message list)."""

    content: str = ""
    messages: list[dict] | None = None
    app_id: str | None = None
    user_id: str
    agent_id: str = ""
    session_id: str = ""
    memory_at: int | None = None


class AddResponse(BaseModel):
    """Response for a successful add, including the new memory id and timing."""

    success: bool
    memory_id: str
    request_id: str
    processing_time_ms: float


class SearchRequest(BaseModel):
    """Request body for semantic search with scope and route parameters."""

    query: str
    app_ids: list[str] | None = None
    user_id: str
    agent_ids: list[str] | None = None
    session_ids: list[str] | None = None
    limit: int = 10
    min_score: float = 0.0
    profile_limit: int = -1
    profile_min_score: float = 0.3
    intention_limit: int = 0
    created_after: int | None = None
    debug: bool = False


class SearchResponse(BaseModel):
    """Response carrying grouped search results and timing."""

    success: bool
    request_id: str
    memories: dict
    processing_time_ms: float


class DeleteResponse(BaseModel):
    """Response for deleting a single memory."""

    success: bool


class DeleteBulkResponse(BaseModel):
    """Response for a bulk delete, including the number removed."""

    success: bool
    deleted: int = 0
    error_code: int | None = None


class UpdateRequest(BaseModel):
    """Request body for updating a memory's content."""

    content: str


class UpdateResponse(BaseModel):
    """Response for a successful update."""

    success: bool
    memory_id: str | None = None
    error_code: int | None = None


class DigestResponse(BaseModel):
    """Response for System2 digest."""

    success: bool
    processed: int = 0
    cores_created: int = 0


class ScopeSummary(BaseModel):
    """One tenant scope row."""

    app_id: str
    user_id: str
    agent_id: str = ""
    memory_count: int = 0


class CapabilitiesResponse(BaseModel):
    """Tool manifest for REST clients and npm MCP code generation."""

    sdk_version: str
    mode: str
    tools: list[dict]
    openapi_url: str


class ErrorResponse(BaseModel):
    """Contract-aligned error envelope."""

    success: bool = False
    error_code: int
    error_message: str


class HealthResponse(BaseModel):
    """Health-check response with serving status and uptime."""

    status: str
    message: str
    uptime_seconds: float


class PingResponse(BaseModel):
    """Ping response with server time and timestamp."""

    message: str
    server_time: str
    timestamp: int


class InfoResponse(BaseModel):
    """Service info response (SDK version, mode, build)."""

    sdk_version: str
    mode: str
    build: str
    capabilities_url: str


class MemoryItem(BaseModel):
    """A single memory record as returned by list/get endpoints."""

    memory_id: str
    content: str
    category: str
    tags: list[str] = Field(default_factory=list)
    memory_at: int | None = None
    gmt_created: int | None = None
    gmt_modified: int | None = None
