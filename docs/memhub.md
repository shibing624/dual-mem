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

服务器上只跑一个 Core，监听它自己的 `127.0.0.1:8765`。你的电脑用 SSH 把这个端口接到本机，再告诉本机的工具「记忆服务在本机这个端口」。完整步骤（服务器安装、token、端口被占用、断开之后回到本机库）见 [在自己的服务器上跑 MemHub](memhub-server.md)。

```bash
# 本机 8766 → 服务器的 127.0.0.1:8765。-N 表示只转发，不打开远程 shell。
ssh -N -L 8766:127.0.0.1:8765 user@memory-host

# api-key 是服务器上 ~/.dual-mem/runtime-token 的内容，不是模型的 API key。
dual-mem hub attach --base-url http://127.0.0.1:8766 --api-key 'runtime-token'
```

不想连了：`dual-mem hub detach`，这台电脑恢复使用本机记忆。服务器上的那一份还在。

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
- `install-host` 会安装 MCP、Skill，以及该工具可用的 capture / recall Hook。

### 写入流程

- Core 收到原始 Evidence 后先写入 `hub.db`，随即返回回执。之后由唯一的 worker 按事件时间（`memory_at`）串行调用 `MemoryClient`，所以扫描和 Hook 都不用等 LLM。
- 模型调用失败会按退避自动重试；重试用尽的写入可以用 `dual-mem hub retry` 重新排队。
- 会话扫描游标存在发起扫描那台机器的 `cursors.db`，只越过 Core 已接收的批次；Core 不可达时游标不动。
- 冷启动可以先 `scan --raw-only`，只存 L0 原文、不调抽取模型；之后 `dual-mem hub extract` 在后台补跑抽取，复用已存的 L0。
- 普通扫描按会话分批写入，`session_id` 带来源前缀，例如 `codex:<session-id>`。

### 云端单实例

- 云模式仍然只有一个数据库、一个写者。Core 只监听云主机的 loopback，客户端用 SSH、Tailscale 或 WireGuard 隧道映射到自己的 loopback。
- 远程 Core 的密钥在它自己的 `~/.dual-mem/runtime-token`。`attach` 会把显式连接保存在本机的 `hub.json`。
- 会话解析和游标留在客户端；记忆内容、Evidence、上传目录和向量 / 图数据库只存在 Core 主机的 `~/.dual-mem`。
- MemHub 不做 `push` / `pull`、快照合并或网络数据库复制。不走 loopback 隧道时，远程 `base_url` 必须是 HTTPS。
- 离线可用、联网自动合并的多设备同步在 Roadmap 中。

### 其他命令

```bash
dual-mem hub hook-search            # 供宿主 Hook 调用：搜索并输出注入块
dual-mem hub capture claude-code    # 供宿主 Hook 调用：上报当前会话
```
