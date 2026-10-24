# MemHub：跨 AI 工具共享你的个人记忆

在 Cursor 里交代过的偏好，Codex 并不知道；在 Claude Code 里定下的规范，到了 Hermes 又要重新说明。

**MemHub 让你的 AI 工具共享一份个人记忆。** 记忆是你的长期资产，不被任何单一工具锁定，每次开口都从懂你开始。

```text
   Claude Code   Codex   Cursor   Hermes   Agentica
        \          |        |        |       /
         ─────────────  MemHub  ─────────────
                         │
                你的个人记忆（本机 / 你的云主机）
```

## 你会得到什么

- **说一次，处处生效**：在一个工具里提过的偏好、规范和项目背景，其他工具自动知道。
- **零配置**：工具自动发现本机的记忆服务，不用填地址，也不用复制密钥。
- **不拖慢工作**：对话先存下来立刻返回，整理在后台进行。
- **历史不浪费**：已有的会话记录一键导入，第一天就"懂你"。
- **数据归你**：默认只存在你的电脑上；也可以放到你自己的云主机，让多台电脑连同一份记忆。

当前支持：**Claude Code、Codex、Cursor、Hermes、Agentica**。OpenCode、WorkBuddy 等更多工具正在接入，见 [Roadmap](https://github.com/shibing624/dual-mem#roadmap)。

## 三步开始

```bash
pip install "dual-mem[all]"

# 1. 启动记忆服务（登录后自动运行）
dual-mem hub install-service

# 2. 接入本机的 AI 工具。all 会写入下面五个的配置；只用其中几个就写出名字
dual-mem hub install-host all
# dual-mem hub install-host claude-code
# dual-mem hub install-host codex

# 3. 导入过去的对话
dual-mem hub scan --source all
```

然后就可以问了：

```bash
dual-mem hub recall "我之前做过哪些技术选择"
dual-mem hub status
```

> 自启服务不继承 shell 环境变量，模型 API key 请写在 `~/.dual_mem/config.yaml`。

## 多台电脑共用一份记忆

服务器上只跑一个 Core，监听它自己的 `127.0.0.1:8765`。每台电脑一条命令连上：

```bash
dual-mem hub connect user@memory-host
```

它通过 SSH 读服务器的 `runtime-token`，在本机空闲端口建常驻 SSH 隧道（登录服务，断线重连），确认身份后写好本机 `hub.json`。Core 仍只监听服务器的 loopback。完整步骤（服务器安装、免密 SSH、自定义端口、不用 SSH 时怎么接）见 [在自己的服务器上跑 MemHub](memhub-server.md)。

不想连了：`dual-mem hub detach`，同时卸载隧道服务，这台电脑恢复使用本机记忆。服务器上的那一份还在。

### 记忆是怎么"同步"的

严格说没有同步：**任何时候只有一个 Core、一份库**，各个工具只是往里送对话、从里取记忆。

```text
┌─────────── 你的电脑 ────────────┐                    ┌───── Core 所在机器 ─────┐
│ Claude Code ┐                   │                    │                        │
│ Codex       ├ capture Hook ─────┼──┐                 │  dual-mem hub serve    │
│ Agentica    ┘ （一轮对话结束）    │  │  本机 loopback   │  （唯一写者）            │
│                                 │  ├─ 或 SSH 隧道 ───┼─▶ 127.0.0.1:8765        │
│ 你输入提示词 ─ hook-search ─────┼──┘                 │  ~/.dual-mem/{hub.db,  │
│ Agent 主动查 ─ MCP memory_* ────┼────────────────────┼─▶          store}       │
└─────────────────────────────────┘                    └────────────────────────┘
```

1. **写**：一轮对话结束，宿主触发 capture Hook（Claude Code `Stop`、Agentica `run.completed`、Codex `Stop`）。它按本机 `cursors.db` 的游标读出会话日志里新增的轮次，交给 Core 后立刻返回，不等 LLM。
2. **整理**：Core 先把 Evidence 写进持久队列 `hub.db`，再由唯一 worker 在后台抽取、JEV 裁决、写入记忆；dual 模式下空闲时自动 `digest()`。
3. **读**：你提交提示词时，recall Hook（`hook-search`）把相关记忆注入上下文；Agent 也可以通过 MCP 工具（`memory_search` 等）主动查。

默认 Core 跑在本机，同一台电脑上的所有工具共享它。`hub connect` 之后 Core 换到服务器，**结构不变**：本机的 Hook 和 MCP 改为经 SSH 隧道访问服务器上的 Core，本机只留游标和连接配置。多台电脑 `connect` 同一台服务器，就是同一份记忆。断网时提交失败、游标不前进，恢复后下一次 capture 自动补交。

## 常用命令

| 想做什么 | 命令 |
|---|---|
| 只接入某个工具 | `dual-mem hub install-host cursor` |
| 取消接入 | `dual-mem hub uninstall-host cursor` |
| 只导入某个工具的历史 | `dual-mem hub scan --source codex` |
| 快速导入、稍后再整理 | `dual-mem hub scan --source all --raw-only`，之后 `dual-mem hub extract` |
| 重试失败的整理任务 | `dual-mem hub retry` |
| 停止开机自启 | `dual-mem hub uninstall-service` |

---

## 架构、安全模型与写入流程

### 单 Core 架构

MemHub 是本地优先的单 Core 架构，不是多机数据库同步：

```text
CLI transcript / MCP / Hook
        ↓ loopback HTTP
dual-mem hub serve（唯一写者）
        ↓
~/.dual-mem/{hub.db,store}
```

前台调试用 `dual-mem hub serve`；日常用 `install-service`（macOS LaunchAgent / Linux systemd --user）。

### 发现与鉴权

- Core 只监听 `127.0.0.1:8765`，启动时自动生成 `runtime-token` 并发布 `runtime.json`，同机的 CLI、MCP 和 Hook 自动发现，不需要配置。
- Bearer Token 只保存在权限为 `0600` 的 `~/.dual-mem/runtime-token`，不会复制到各宿主的配置里。
- `install-host` 会安装 MCP、Skill，以及该工具可用的 capture / recall Hook。改写 YAML 配置（Agentica、Hermes）时保留原有注释、引号与缩进；`uninstall-host` 后文件恢复原样。
- MemHub 自带的 MCP 服务（`python -m dual_mem.memhub.mcp`，由 `install-host` 写入宿主配置）提供 `memory_search`、`memory_add`、`conversation_search`、`memory_status`，请求同样走 `hub.json` / 本机发现的 Core。需要 mcp 2.x（`pip install "dual-mem[mcp]"`）。

### 写入流程

- Core 收到原始 Evidence 后先写入 `hub.db`，随即返回回执。之后由唯一的 worker 按事件时间（`memory_at`）串行调用 `MemoryClient`，所以扫描和 Hook 都不用等 LLM。
- 模型调用失败会按退避自动重试；重试用尽的写入可以用 `dual-mem hub retry` 重新排队。
- 会话扫描游标存在发起扫描那台机器的 `cursors.db`，只越过 Core 已接收的批次；Core 不可达时游标不动。
- 冷启动可以先 `scan --raw-only`，只存 L0 原文、不调抽取模型；之后 `dual-mem hub extract` 在后台补跑抽取，复用已存的 L0。
- 普通扫描按会话分批写入，`session_id` 带来源前缀，例如 `codex:<session-id>`。
- Claude Code / Codex / Agentica 会话里的工具调用和结果随对话一起送达（单条结果过长时保留头尾），会话的工作目录和 git 分支作为 `workspace_id` / `branch`。Core 据此在对话记忆之外沉淀 procedure 工程经验；`recall` 和 `hook-search` 默认返回最多 3 条相关经验，在注入块里标为 `[procedure]`。
- Core 以 dual 模式运行时（`hub serve --mode dual`），写入队列空闲 `--digest-idle` 秒（默认 300，0 = 关闭）后自动 `digest()`，不需要任何人手动触发。

### 云端单实例

- 云模式仍然只有一个数据库、一个写者。Core 只监听云主机的 loopback，客户端用 SSH、Tailscale 或 WireGuard 隧道映射到自己的 loopback。
- 远程 Core 的密钥在它自己的 `~/.dual-mem/runtime-token`。`connect`（或手动的 `attach`）把连接保存在本机的 `hub.json`。
- 会话解析和游标留在客户端；记忆内容、Evidence、上传目录和向量 / 图数据库只存在 Core 主机的 `~/.dual-mem`。
- MemHub 不做 `push` / `pull`、快照合并或网络数据库复制。不走 loopback 隧道时，远程 `base_url` 必须是 HTTPS。
- 离线可用、联网自动合并的多设备同步在 Roadmap 中。

### 其他命令

```bash
dual-mem hub hook-search            # 供宿主 Hook 调用：搜索并输出注入块
dual-mem hub capture claude-code    # 供宿主 Hook 调用：上报当前会话
```
