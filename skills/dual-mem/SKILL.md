---
name: dual-mem
description: >
  通过 dual-mem 为 agent 管理长期记忆：写入、检索、更新、删除与 scope 管理。
  触发场景：用户透露稳定个人信息/偏好、陈述值得长期记住的事实或计划，或对话需要
  回忆该用户过去说过什么时。使用 MCP 工具或 CLI（dual-mem）调用 MemoryClient 方法。
---

# dual-mem 记忆技能

dual-mem 把对话沉淀为**分层长期记忆**（L0–L7），检索时按 profile / proactive / normal 三路返回。
它管**跨会话记忆**，不能替代 Agent 自己的当前窗口（WorkingMemory）。

## 工具 ↔ SDK 对照

| MCP 工具 | CLI | REST | MemoryClient |
|---|---|---|---|
| `memory_add` | `dual-mem add` | `POST /v1/memories/` | `add` |
| `memory_search` | `dual-mem search` | `POST /v1/memories/search` | `search` |
| `memory_list` | `dual-mem list` | `GET /v1/memories/` | `list` |
| `memory_get` | `dual-mem get <id>` | `GET /v1/memories/{id}` | `get` |
| `memory_update` | `dual-mem update` | `PUT /v1/memories/{id}` | `update` |
| `memory_delete` | `dual-mem delete` | `DELETE /v1/memories/{id}` | `delete` |
| `memory_delete_scope` | `dual-mem delete-scope` | `DELETE /v1/memories/?confirm=true` | `delete_bulk` |
| `memory_list_scopes` | `dual-mem list-scopes` | `GET /v1/scopes/` | `list_scopes` |
| `memory_digest` | `dual-mem digest` | `POST /v1/digest/` | `digest` |

发现：`GET /v1/capabilities` 返回完整 HTTP 映射（供 npm MCP codegen）。

## 归属标识（scope）

每次读写必须一致：

- **`app_id`** — 应用/产品命名空间（如 `"agentica"`）
- **`user_id`** — 终端用户 ID
- 可选 **`agent_id`** / **`session_id`** — 多 Bot 或会话级隔离

## 写入方式（用户自选，SDK 不强制）

| 方式 | API | 适用 | 代价 |
|------|-----|------|------|
| 单条实时 | `add(content=...)` | 合规/关键事实需立刻落库 | 每次 ~2 次 LLM + 若干 embed（约 10–15s） |
| 多轮批量 | `add(messages=[...])` | 普通助手、示例默认、会话末沉淀 | 同上但 **1 次** pipeline，上下文更完整 |

**建议（非约束）**：Agent 集成可「每轮 search、会话结束 add(messages)」以省成本；用户若需 mid-session 单条写入，完全合法。

CLI：`--content` 或 `--messages-file` / `--messages-json` 二选一。

## 何时写入（memory_add）

- 稳定画像：身份、职业、所在地、长期偏好
- 值得记住的事实或计划
- 用户纠正旧信息（会形成演化链）

不要写：纯寒暄、仅当前任务用的临时上下文。

## 何时检索（memory_search）

- 需要个性化背景（「按我习惯推荐」）
- 用户引用过去（「我之前说的那个项目」）

## CLI 示例

单条：

```bash
dual-mem add --content "用户在深圳做 ML" --app-id agentica --user-id u1
```

会话结束批量写入多轮对话：

```bash
# messages.json: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}, ...]
dual-mem add --messages-file messages.json --app-id agentica --user-id u1

# 或内联 JSON
dual-mem add --messages-json '[{"role":"user","content":"我搬到北京了"}]' \
  --app-id agentica --user-id u1
```

检索与管理：

```bash
dual-mem search "用户做什么工作" --app-id agentica --user-id u1
dual-mem list --app-id agentica --user-id u1
dual-mem get <memory_id>
dual-mem update <memory_id> --content "新内容"
dual-mem delete <memory_id>
dual-mem delete-scope --app-id agentica --user-id u1 --confirm
dual-mem list-scopes --app-id agentica
dual-mem digest   # dual 模式
```

## MCP 示例

```text
memory_add(app_id="agentica", user_id="u1", content="...")
memory_add(app_id="agentica", user_id="u1", messages=[{"role":"user","content":"..."}, ...])
memory_search(query="...", app_ids=["agentica"], user_id="u1")
memory_list(app_id="agentica", user_id="u1")
memory_get(memory_id="...")
memory_update(memory_id="...", content="...")
memory_delete(memory_id="...")
memory_delete_scope(app_id="agentica", user_id="u1", confirm=true)
memory_list_scopes(app_id="agentica")
memory_digest()
```

启动：`dual-mem-mcp`（stdio）或 `dual-mem-mcp --transport streamable-http --port 8765`

## 检索结果解读

- **profile** — 画像 / 身份 / 长期偏好（优先用）
- **proactive** — 推断意图（dual + `intention_limit>0`）
- **normal** — 普通事实与知识

**evolution_chain**：同一记忆多次更新时，按最新→最旧排列；`[0]` 为当前版本。

## 模式

- **system1**（默认）：短期记忆，写入即可检索
- **dual**：+ System2 深度记忆（L6 行为模式、L7 意图）；`memory_digest` 或自动后台巩固
