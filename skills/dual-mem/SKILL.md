---
name: dual-mem
description: >
  通过 dual-mem 分层记忆 SDK 为 agent 写入与检索长期记忆。
  触发场景：用户透露了稳定的个人信息/偏好/身份（"我在深圳做 ML"、"我不吃辣"）、
  陈述了值得长期记住的事实或计划，或当前对话需要回忆该用户过去说过的内容
  （"我之前提过的那个项目"、"按我的习惯来"）时。用 CLI（`dual-mem`）或 MCP 工具
  （memory_add / memory_search 等）写入与检索，返回结果按 profile/proactive/normal
  三路分组，演化记忆带 evolution_chain 历史版本链。
---

# dual-mem 记忆技能

dual-mem 是分层记忆 SDK：把对话中值得长期保留的信息抽取、去重、演化后存起来，
之后用语义检索召回。它**不是**临时草稿纸——只存对未来对话有用的稳定信息。

## 何时写入（add）

- 用户透露稳定画像：身份、职业、所在地、长期偏好、过敏/禁忌、家庭情况。
- 用户陈述值得记住的事实或计划："我下个月要去新西兰自驾"。
- 用户纠正了之前的信息（会自动形成演化链，不要怕覆盖）。

不要写入：一次性闲聊、当前任务里的临时上下文、可由当前对话直接得到的信息。

## 何时检索（search）

- 回答前需要个性化背景时（"按我习惯推荐"）。
- 用户引用过去的内容（"我之前说的那个"）。
- 想确认是否已知某用户的某项信息。

## CLI 调用范式

写入：

```bash
dual-mem add --content "用户叫张三，在深圳做 ML" --app-id myapp --user-id user_001
```

检索（结果已按三路分组格式化）：

```bash
dual-mem search "用户做什么工作" --app-id myapp --user-id user_001 --limit 5
```

其它：`dual-mem list --app-id myapp --user-id user_001`、
`dual-mem get <memory_id>`、`dual-mem delete <memory_id>`、
`dual-mem digest`（ultra 模式触发后台沉淀）。
启动服务：`dual-mem serve --host 0.0.0.0 --port 8000`（REST）、
`dual-mem-mcp`（MCP stdio，供 Cursor/Claude Desktop 经 uvx 拉起）、
`dual-mem-mcp --transport streamable-http --port 8765`（MCP over HTTP，暴露 `/mcp`）。

## MCP 工具范式

通过 MCP 接入时使用：`memory_add(content, app_id, user_id, ...)`、
`memory_search(query, app_ids, user_id, ...)`、`memory_get`、`memory_list`、`memory_delete`。
`app_id` × `user_id`（× 可选 `agent_id` / `session_id`）唯一确定记忆归属，检索时务必传一致的标识。

## 检索结果三路分组

search 返回 `memories` 对象，固定三个 key，按重要性顺序使用：

- **profile**：稳定的用户画像 / 身份 / 长期模式（最优先呈现）。
- **proactive**：推断出的用户意图（仅 ultra 模式非空，pro/lite 为空）。
- **normal**：普通事实与知识记忆。

渲染进上下文时按 profile → proactive → normal 顺序拼装。

## 演化链解读（evolution_chain）

一条记忆若被多次更新，会带 `evolution_chain` 字段，按 **最新 → 最旧** 排序：
`evolution_chain[0]` 就是当前最新版本（content 与外层一致），其余是历史版本。
据此可以理解信息的变化轨迹（如"用户从 Java 转向 Python"）。展示时注意去重：
要么只渲染演化链，要么跳过外层 content 从链里组装，避免最新版本重复出现两次。

## lite / pro / ultra 区别

- **lite**：只做写入与语义检索，不调用 LLM，零成本、最快。proactive 路始终为空。
- **pro**：写入路径中由 LLM 同步完成抽取 / 整理 / 去重，无后台 worker。
- **ultra**：完整 System1 + System2。写入同步返回后，System2 在后台异步做反思、
  抽象与演化链构建；可调 `digest` 主动触发沉淀。proactive（意图）路在此模式下才有内容。
