# MemHub

MemHub 让 Agentica、Claude Code、Codex、Cursor 和 Hermes 共享一个个人记忆库。
它是本地优先的单 Core 架构，不是多机数据库同步：

```text
CLI transcript / MCP / Hook
        ↓ loopback HTTP
dual-mem hub serve（唯一写者）
        ↓
~/.dual-mem/{hub.db,store,uploads}
```

## 本机使用

先启动唯一 Core：

```bash
dual-mem hub serve
```

Core 只监听 `127.0.0.1:8765`，自动生成 `runtime-token` 并发布
`runtime.json`。同机 CLI、MCP 和 Hook 会自动发现，不需要配置 URL 或密钥。
在另一个终端直接安装和使用：

```bash
dual-mem hub install-host all
dual-mem hub scan --source all
dual-mem hub status
dual-mem hub recall "我之前做过哪些技术选择"
```

`install-host` 安装 MCP、Skill 和可用的 capture/recall Hook。Bearer Token
只保存在权限为 `0600` 的 `~/.dual-mem/runtime-token`，不会复制到各宿主配置。

Core 把收到的原始 Evidence 先写入 `hub.db`，再由唯一 worker 串行调用
`MemoryClient`。会话扫描游标位于发起扫描机器的 `cursors.db`；网络失败时不会
越过尚未提交的 transcript。

## 云上单实例

云模式仍然只有一个数据库和一个写者。Core 只监听云主机 loopback，客户端用
SSH、Tailscale 或 WireGuard 隧道把它映射到自己的 loopback：

```bash
ssh -N -L 8765:127.0.0.1:8765 user@memory-host

dual-mem hub attach \
  --base-url http://127.0.0.1:8765 \
  --api-key 'your-secret'
```

远程 Core 的密钥位于其 `~/.dual-mem/runtime-token`。`attach` 是高级功能，会把
显式连接保存在本机 `hub.json`；`dual-mem hub detach` 删除它并恢复本地自动发现。

会话解析和游标仍在客户端，记忆内容、Evidence、上传目录和向量/图数据库只存在
Core 主机的 `~/.dual-mem`。MemHub 不执行 `push`、`pull`、快照合并或网络数据库
复制。若不使用 loopback 隧道，远程 `base_url` 必须是 HTTPS。

## 常用命令

```bash
dual-mem hub scan --source codex
dual-mem hub scan --source all --raw-only
dual-mem hub install-host cursor
dual-mem hub uninstall-host cursor
dual-mem hub hook-search
dual-mem hub capture claude-code
```

`--raw-only` 只保存 L0 Evidence，不调用抽取模型。普通扫描按会话分批写入，并以
来源 CLI 前缀保存 `session_id`，例如 `codex:<session-id>`。
