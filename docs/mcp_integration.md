# MCP 接入与部署

dual-mem 的 MCP server 把记忆能力暴露为 agent 可调用的工具，基于 FastMCP 实现，
**同时支持两种传输**：

- **stdio**：本地由 Cursor / Claude Desktop 直接拉起进程（最常用）。
- **streamable-http**：作为独立 HTTP 服务运行，暴露 `/mcp` 端点，供远程/多客户端接入。

底层都复用同一个 `MemoryClient`（配置走 `~/.dual_mem/config.yaml`，见 `config.example.yaml`）。

## 暴露的工具

| 工具 | 说明 |
|---|---|
| `memory_add(content, app_id, user_id, agent_id?, session_id?)` | 写入一条记忆，返回 `memory_id` |
| `memory_search(query, app_ids, user_id, agent_ids?, limit?, min_score?, intention_limit?)` | 语义检索，结果按 profile/proactive/normal 三路分组；演化过的记忆带 `evolution_chain`（最新→最旧） |
| `memory_get(memory_id)` | 取单条，不存在返回 null |
| `memory_list(app_id, user_id, agent_id?, limit?)` | 列出某 app/user（可选 agent）下的 ACTIVE 记忆 |
| `memory_delete(memory_id)` | 删除单条（幂等） |

> 默认 `intention_limit=0` 时 proactive 路恒空；需要主动意图召回时显式传正整数（仅 dual 模式有 L7 意图）。`min_score` 默认 0.4，约束 normal 路。
>
> dual 模式的 System2 沉淀触发（`digest`）**未**作为 MCP 工具暴露，按需通过 SDK 或 CLI（`dual-mem digest`）触发；`per_write` / `scheduled` 触发模式则由 SDK 内部自动调度。

## 启动方式

### 1. stdio（本地）

```bash
dual-mem-mcp                      # 等价 --transport stdio
# 或经 SDK CLI：
dual-mem mcp
```

### 2. Streamable HTTP（暴露 /mcp）

```bash
dual-mem-mcp --transport streamable-http --host 0.0.0.0 --port 8765
# 端点：http://<host>:8765/mcp
```

### 3. 经 uvx 免安装运行

发布到 PyPI 后：

```bash
uvx dual-mem-mcp                                   # stdio
uvx dual-mem-mcp --transport streamable-http       # HTTP
```

本地源码（未发布）可用 `--from`：

```bash
uvx --from /Users/xuming/Documents/Codes/dual-mem dual-mem-mcp
```

## 客户端配置

### Cursor（stdio + uvx）

`~/.cursor/mcp.json`（或项目内 `.cursor/mcp.json`）：

```json
{
  "mcpServers": {
    "dual-mem": {
      "command": "uvx",
      "args": ["dual-mem-mcp"]
    }
  }
}
```

未发布时用本地源码：

```json
{
  "mcpServers": {
    "dual-mem": {
      "command": "uvx",
      "args": ["--from", "/Users/xuming/Documents/Codes/dual-mem", "dual-mem-mcp"]
    }
  }
}
```

或直接用已安装的入口（`pip install -e .` 后）：

```json
{
  "mcpServers": {
    "dual-mem": { "command": "dual-mem-mcp", "args": [] }
  }
}
```

### Claude Desktop（stdio + uvx）

`claude_desktop_config.json`（macOS：`~/Library/Application Support/Claude/`）：

```json
{
  "mcpServers": {
    "dual-mem": {
      "command": "uvx",
      "args": ["dual-mem-mcp"]
    }
  }
}
```

### Streamable HTTP 客户端

先以 HTTP 模式起服务，再在支持 HTTP MCP 的客户端里填端点 `http://127.0.0.1:8765/mcp`。
握手符合 MCP Streamable HTTP 规范（`initialize` 走 `POST /mcp`，返回 `mcp-session-id`）。

## 配置来源

MCP server 启动时用 `Settings()` 读取配置，优先级：

```
显式传参  >  DUAL_MEM_* 环境变量  >  ~/.dual_mem/config.yaml  >  默认值
```

把 `llm_api_key` / `embed_api_key` / `mode` 等写进 `~/.dual_mem/config.yaml` 即可，
无需在 MCP 客户端配置里塞密钥。

## 与 Skill 的关系

`skills/dual-mem/SKILL.md` 是给 agent 的使用说明（何时写入/检索、如何解读三路结果与演化链）。
推荐链路：**Skill 指导 agent → 经 MCP 调工具 → dual-mem SDK → 存储**。
