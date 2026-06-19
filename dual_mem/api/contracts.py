# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Machine-readable contract for memory tools — shared by REST, Python MCP,
and future npm/TypeScript MCP (HTTP transport). Each entry maps an operation name to its
HTTP method/path; MCP tool names match ``name`` exactly.
"""
from typing import Any

# Stable tool ids — keep in sync with MemoryOperations + mcp/server.py
MEMORY_TOOL_CONTRACTS: list[dict[str, Any]] = [
    {
        "name": "memory_add",
        "summary": "Write a memory (content or messages)",
        "http": {"method": "POST", "path": "/v1/memories/"},
    },
    {
        "name": "memory_search",
        "summary": "Semantic search (profile / proactive / normal)",
        "http": {"method": "POST", "path": "/v1/memories/search"},
    },
    {
        "name": "memory_list",
        "summary": "List ACTIVE memories in a scope",
        "http": {"method": "GET", "path": "/v1/memories/"},
    },
    {
        "name": "memory_get",
        "summary": "Get one memory by id",
        "http": {"method": "GET", "path": "/v1/memories/{memory_id}"},
    },
    {
        "name": "memory_update",
        "summary": "Replace content and re-embed",
        "http": {"method": "PUT", "path": "/v1/memories/{memory_id}"},
    },
    {
        "name": "memory_delete",
        "summary": "Delete one memory by id",
        "http": {"method": "DELETE", "path": "/v1/memories/{memory_id}"},
    },
    {
        "name": "memory_delete_scope",
        "summary": "Bulk delete by app_id scope (confirm required)",
        "http": {"method": "DELETE", "path": "/v1/memories/"},
    },
    {
        "name": "memory_list_scopes",
        "summary": "List distinct app_id + user_id + agent_id scopes",
        "http": {"method": "GET", "path": "/v1/scopes/"},
    },
    {
        "name": "memory_digest",
        "summary": "Run System2 consolidation (dual mode)",
        "http": {"method": "POST", "path": "/v1/digest/"},
    },
]
