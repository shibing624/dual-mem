from pydantic import BaseModel, Field


class AddRequest(BaseModel):
    content: str = ""
    messages: list[dict] | None = None
    app_id: str
    user_id: str
    agent_id: str = ""
    session_id: str = ""
    mode: str | None = None
    memory_at: int | None = None


class AddResponse(BaseModel):
    success: bool
    memory_id: str
    request_id: str
    processing_time_ms: float


class SearchRequest(BaseModel):
    query: str
    app_ids: list[str]
    user_id: str
    agent_ids: list[str] | None = None
    session_ids: list[str] | None = None
    limit: int = 10
    min_score: float = 0.0
    profile_limit: int = -1
    profile_min_score: float = 0.3
    created_after: int | None = None


class SearchResponse(BaseModel):
    success: bool
    request_id: str
    memories: dict
    processing_time_ms: float


class DeleteResponse(BaseModel):
    success: bool


class DeleteBulkResponse(BaseModel):
    success: bool
    deleted: int


class ErrorResponse(BaseModel):
    success: bool = False
    error_code: int
    error_message: str


class HealthResponse(BaseModel):
    status: str
    message: str
    uptime_seconds: float


class PingResponse(BaseModel):
    message: str
    server_time: str
    timestamp: int


class InfoResponse(BaseModel):
    sdk_version: str
    mode: str
    build: str


class MemoryItem(BaseModel):
    memory_id: str
    content: str
    category: str
    tags: list[str] = Field(default_factory=list)
    memory_at: int | None = None
    gmt_created: int | None = None
    gmt_modified: int | None = None
