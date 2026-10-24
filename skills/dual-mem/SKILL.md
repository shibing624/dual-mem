---
name: dual-mem
description: >
  通过 dual-mem 为 agent 管理长期记忆：写入、检索、更新、删除与 scope 管理。
  触发场景：用户透露稳定个人信息/偏好、陈述值得长期记住的事实或计划，或对话需要
  回忆该用户过去说过什么时。使用 MCP 工具或 CLI（dual-mem）调用 MemoryClient 方法。
---

# dual-mem 记忆技能

dual-mem 把对话沉淀为四层长期记忆（`L0_RAW` / `L1_FACT` / `L2_PROFILE` / `L3_DERIVED`），检索时按 profile / proactive / normal 三路返回。
它管**跨会话记忆**，不能替代 Agent 自己的当前窗口（WorkingMemory）。

dual-mem 原生对接 5 类社区生态，统一入口在 `dual_mem.integrations`：

| 生态 | 接入方式 | 关键文件 / 命令 |
|------|----------|----------------|
| skill | 本 SKILL.md + `MemoryClient` | `skills/dual-mem/SKILL.md` |
| mcp | MCP server（mcp 2.x `MCPServer`，`dual-mem-mcp`，stdio / streamable-http） | `dual_mem/mcp/server.py` |
| hermes agent | `DualMemHermesProvider` 原生插件（被动 prefetch + 异步 sync_turn） | `dual_mem/integrations/hermes.py` |
| openclaw | `DualMemOpenClawProvider` 原生插件（同源 MemoryProvider 契约） | `dual_mem/integrations/openclaw.py` |
| agentica | `DualMemMemory` 记忆模块（`add_messages` / `retrieve`）+ `DualMemWorkspace`（`Agent(workspace=…)` 直接接入） | `dual_mem/integrations/agentica.py` |
| claude code | `dual-mem-hook` 子命令（UserPromptSubmit / Stop） | `dual_mem/integrations/claude_code.py` |

所有适配器都基于内嵌 `MemoryClient`（不再依赖外部 one-memory server），配置走
`DUAL_MEM_*` 环境变量 / `~/.dual-mem/config.yaml`。程序化构造：

```python
from dual_mem.integrations import get_backend
mcp_server = get_backend("mcp")                       # MCPServer 实例
hermes = get_backend("hermes")                        # 注册进 Hermes runtime
openclaw = get_backend("openclaw")
memory = get_backend("agentica", user_id="u1")        # agentica 记忆模块
```

### agentica 接入

agentica 的 `Agent` 通过 `workspace` 对象消费长期记忆。dual-mem 提供两层接入，
均无需改动 agentica 框架代码：

- **`DualMemMemory`（记忆模块）**：暴露 `add_messages` / `retrieve`，用于手动把召回
  结果注入 system prompt。
- **`DualMemWorkspace`（Workspace 适配器）**：组合原生 agentica `Workspace`（承接
  AGENTS.md / PERSONA.md / USER.md / skills 等文件型上下文），仅把「语义记忆」相关方法
  覆盖为调用 dual-mem。可直接 `Agent(workspace=...)`，自动召回 + 自动保存。

> 依赖 agentica：`pip install 'dual-mem[agentica]'`（或 `pip install agentica`）。
> 凭证走 `DUAL_MEM_*` 环境变量 / `~/.dual-mem/config.yaml`，或构造时显式传入 `client=`。

**方式一：作为 Agent 的原生长期记忆（推荐）**

```python
from dual_mem.integrations.agentica import get_agentica_memory_backend
from agentica import Agent

ws = get_agentica_memory_backend(
    user_id="u1",                 # 或 client=<已构造的 MemoryClient>
    workspace_dir=".agentica",    # 文件型上下文（AGENTS.md 等）落盘目录
)
agent = Agent(workspace=ws, enable_long_term_memory=True)
# Agent 的 system prompt 会自动注入 dual-mem 召回；
# 内置 BuiltinMemoryTool 的 save_memory / search_memory 也会落到 dual-mem。
```

等价地，从 `DualMemMemory` 派生 workspace：

```python
memory = get_backend("agentica", user_id="u1")
ws = memory.as_workspace(workspace_dir=".agentica")
agent = Agent(workspace=ws, enable_long_term_memory=True)
```

**方式二：手动注入（更可控）**

```python
memory = get_backend("agentica", user_id="u1")

# 存：把一轮多轮对话沉淀为记忆
await memory.add_messages(
    [{"role": "user", "content": "我喜欢燕麦拿铁"}], user_id="u1"
)

# 取：检索并格式化为 <relevant-memories> 注入块
ctx = await memory.retrieve("用户喜欢什么咖啡?", user_id="u1")
await agent.aprint_response("推荐一杯咖啡", context=ctx)
```

`retrieve` / `add_messages` 均为异步；`get_agentica_memory_backend` 与 `as_workspace`
返回的对象在 `Agent` 内部被 await，故同步上下文里直接用 `asyncio.run` 驱动即可。

## 工具 ↔ SDK 对照

| MCP 工具 | CLI | REST | MemoryClient |
|---|---|---|---|
| `memory_add` | `dual-mem add` | `POST /v1/memories/` | `add` |
| `memory_search` | `dual-mem search` | `POST /v1/memories/search` | `search` |
| `conversation_search` | `dual-mem search-conversation` | `POST /v1/memories/search_conversation` | `search_conversation` |
| `memory_list` | `dual-mem list` | `GET /v1/memories/` | `list` |
| `memory_get` | `dual-mem get <id>` | `GET /v1/memories/{id}` | `get` |
| `memory_update` | `dual-mem update` | `PUT /v1/memories/{id}` | `update` |
| `memory_delete` | `dual-mem delete` | `DELETE /v1/memories/{id}` | `delete` |
| `memory_delete_scope` | `dual-mem delete-scope` | `DELETE /v1/memories/?confirm=true` | `delete_bulk` |
| `memory_list_scopes` | `dual-mem list-scopes` | `GET /v1/scopes/` | `list_scopes` |
| `memory_digest` | `dual-mem digest` | `POST /v1/digest/` | `digest` |

发现：`GET /v1/capabilities` 返回完整 HTTP 映射（供 npm MCP codegen）。

## 归属标识（scope）

**单产品默认：只传 `user_id` 即可**（`app_id` 省略时用 config 的 `default_app_id`，一般为 `"default"`）。

- **`user_id`** — 终端用户 ID（必填）
- **`app_id`** / **`app_ids`** — 可选；多产品共享同一 Memory 服务时再显式传入
- 可选 **`agent_id`** / **`session_id`** — 多 Bot 或会话级隔离

## 写入方式（用户自选，SDK 不强制）

| 方式 | API | 适用 | 代价 |
|------|-----|------|------|
| 单条实时 | `add(content=...)` | 合规/关键事实需立刻落库 | 1 次生成 + 1 次 JEV 裁决 + embed |
| 多轮批量 | `add(messages=[...])` | 普通助手、示例默认、会话末沉淀 | 同上，但整段对话只走 **1 次** pipeline |
| 先存原文 | `add_raw` 再 `distill` | 先保留原话，之后再抽结构化记忆 | distill 复用已有 `L0_RAW`，不再写第二条原文 |

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
dual-mem add --content "用户在深圳做 ML" --user-id u1
```

会话结束批量写入多轮对话：

```bash
# messages.json: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}, ...]
dual-mem add --messages-file messages.json --user-id u1

# 或内联 JSON
dual-mem add --messages-json '[{"role":"user","content":"我搬到北京了"}]' \
  --user-id u1
```

检索与管理：

```bash
dual-mem search "用户做什么工作" --user-id u1
dual-mem search-conversation "原话" --user-id u1
dual-mem list --user-id u1
dual-mem list --user-id u1 --include-history
dual-mem get <memory_id>
dual-mem update <memory_id> --content "新内容"   # 返回新 id，旧 id 仍可 get
dual-mem delete <memory_id>                     # 归档当前版本，不物理删除
dual-mem delete-scope --user-id u1 --confirm
dual-mem list-scopes
dual-mem digest                                 # 仅 dual 生成 L3_DERIVED
```

## MCP 示例

```text
memory_add(user_id="u1", content="...")
memory_add(user_id="u1", messages=[{"role":"user","content":"..."}, ...])
memory_search(query="...", user_id="u1")
conversation_search(query="...", user_id="u1")
memory_list(user_id="u1", include_history=false)
memory_get(memory_id="...")
memory_update(memory_id="...", content="...")   # 新 id
memory_delete(memory_id="...")                  # 归档
memory_delete_scope(user_id="u1", confirm=true)
memory_list_scopes()
memory_digest()                                 # dual → L3_DERIVED
```

启动：`dual-mem-mcp`（stdio）或 `dual-mem-mcp --transport streamable-http --port 8765`

## 检索结果解读

- **profile** — `L2_PROFILE` 与画像类记忆（优先用）
- **proactive** — `L1_FACT` 里 `kind=intention` 的意图；默认 `intention_limit=0`，这一路为空
- **normal** — `L1_FACT` 事实。默认检索不返回 `L0_RAW`；核对原话用 `conversation_search`

**evolution_chain**：同一记忆多次更新时，按最新→最旧排列；`[0]` 为当前版本。system1 要看到链，检索时传 `include_derived=true`。

`update` 追加新版本并返回**新** `memory_id`。`delete` 把当前版本标成 `is_latest=false`，旧 id 仍可 `get`；对已经不是当前版本的 id 再删会失败。

## 模式

- **system1**（默认）：写入即检索。`digest` 返回 `processed=0`，不生成 `L3_DERIVED`。
- **dual**：写入路径与 system1 相同；显式 `digest` / `memory_digest` 才生成 `L3_DERIVED`。没有自动后台巩固。

构造需要三组凭证：LLM、embedding、JEV（`llm_api_key` / `embed_api_key` / `jev_api_key`，或对应的 `DUAL_MEM_*` 环境变量）。
