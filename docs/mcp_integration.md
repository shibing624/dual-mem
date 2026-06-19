# MCP / REST 接入

dual-mem 面向 Agent 提供 **两条 MCP 接入路径**，工具名与 REST 路由一一对应，共享 `MemoryOperations` 实现层：

| 路径 | 状态 | 适用场景 |
|---|---|---|
| **本地 uvx Python MCP** | ✅ 已实现 | Cursor / Claude Desktop 本机直连，`uvx dual-mem-mcp` 或 `pip install dual-mem[mcp]` |
| **Cloud-hosted HTTP MCP** | 🔜 规划中 | 托管记忆服务 + 远程 Agent；基于 REST API 暴露同一套 `memory_*` 契约 |

**当前**：仅本地 Python MCP Server（stdio / streamable-http）可用。  
**后续**：REST 层（`dual-mem serve` + `GET /v1/capabilities`）已对齐 MCP 工具契约，可直接作为 HTTP MCP 的传输底座——无需重复实现业务逻辑，只需在 REST 之上封装 MCP streamable-http 协议（或 npm/TypeScript 薄客户端）。

```
Agent (Cursor / Claude / 自研)
    │
    ├─ 本地 ──stdio/HTTP──▶ dual-mem-mcp ──▶ MemoryOperations ──▶ MemoryClient
    │
    └─ 云端 ──HTTP MCP──▶ dual-mem REST (/v1/...) ──▶ MemoryOperations ──▶ MemoryClient
                              ↑ 同一套 memory_* 工具名与 JSON 契约
```

## 工具 ↔ HTTP ↔ SDK

| 工具 / 操作 | HTTP | MemoryClient |
|---|---|---|
| `memory_add` | `POST /v1/memories/` | `add` |
| `memory_search` | `POST /v1/memories/search` | `search` |
| `memory_list` | `GET /v1/memories/` | `list` |
| `memory_get` | `GET /v1/memories/{id}` | `get` |
| `memory_update` | `PUT /v1/memories/{id}` | `update` |
| `memory_delete` | `DELETE /v1/memories/{id}` | `delete` |
| `memory_delete_scope` | `DELETE /v1/memories/?confirm=true` | `delete_bulk` |
| `memory_list_scopes` | `GET /v1/scopes/` | `list_scopes` |
| `memory_digest` | `POST /v1/digest/` | `digest` |

发现端点：`GET /v1/capabilities` — 返回 `tools` 数组（含 method/path），供 HTTP MCP / npm codegen 自动生成工具定义。

## 本地 MCP（已实现）

```bash
# uvx 零安装（推荐）
uvx dual-mem-mcp
uvx dual-mem-mcp --transport streamable-http --port 8765

# 或 pip 安装后
dual-mem-mcp
dual-mem-mcp --transport streamable-http --port 8765
dual-mem mcp
```

### Cursor（本地 stdio）

```json
{
  "mcpServers": {
    "dual-mem": { "command": "uvx", "args": ["dual-mem-mcp"] }
  }
}
```

配置：`~/.dual_mem/config.yaml` 或 `DUAL_MEM_*` 环境变量。

## REST / HTTP MCP 底座（已实现 REST，HTTP MCP 待封装）

REST 服务本身已可用，可作为 cloud-hosted HTTP MCP 的后端：

```bash
dual-mem serve --host 0.0.0.0 --port 8000
curl http://localhost:8000/v1/capabilities
curl http://localhost:8000/openapi.json
```

后续 HTTP MCP 接入方式（规划）：

- **自托管**：REST + MCP streamable-http 网关（同一 `memory_*` 工具，Bearer 鉴权走 REST）
- **Cloud-hosted**：托管 `dual-mem serve` 实例，Agent 通过 URL + appkey 连接，无需本机 Python 环境

Cursor 侧预期形态（示意，尚未提供官方 npm 包）：

```json
{
  "mcpServers": {
    "dual-mem-cloud": {
      "url": "https://your-host/mcp",
      "headers": { "Authorization": "Bearer <appkey>" }
    }
  }
}
```

## Skill

`skills/dual-mem/SKILL.md` — 指导 agent 何时读写记忆、如何解读 profile/proactive/normal 与演化链。
