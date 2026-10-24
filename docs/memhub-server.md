# 在自己的服务器上跑 MemHub

多台电脑共用**一份**记忆：服务器上只跑一个 MemHub Core，它是唯一写者。笔记本和台式机不各自保存一份再合并，而是把读写都发给这一个 Core。

这不是 `push` / `pull`，也不是两台机器离线各写、联网再合并。后一种还在 [Roadmap](https://github.com/shibing624/dual-mem#roadmap)。

```text
笔记本上的 Claude Code / Codex / Cursor
        │  本机 127.0.0.1:8765
        │  SSH 隧道（或你自己的 HTTPS 反代）
        ▼
服务器 127.0.0.1:8765
dual-mem hub serve
        ▼
服务器上的 ~/.dual-mem/{hub.db,store}
```

Core 只监听 loopback。它不接受 `0.0.0.0`。明文 HTTP 只允许打到 `127.0.0.1`；要直接写一个公网地址，`base_url` 必须是 `https://`。

## 服务器上

需要 Python 3.11+。模型钥匙写在这台机器的 `~/.dual_mem/config.yaml`（`jev_api_key`、`llm_api_key`、`embed_api_key`，三个地址按你的服务改）。登录自启的服务读不到 shell 里临时 `export` 的变量。

```bash
pip install "dual-mem[all]"

# 登录后一直跑。Linux 是 systemd --user，macOS 是 LaunchAgent。
dual-mem hub install-service
```

云主机关掉 SSH 之后，systemd 用户服务会停。让它留在后台：

```bash
sudo loginctl enable-linger "$USER"
```

前台看日志时用 `dual-mem hub serve`。默认地址是 `127.0.0.1:8765`。

第一次启动会在服务器的 `~/.dual-mem/runtime-token` 写下一把 Bearer key，权限 `0600`。以后重启沿用这一把。查看：

```bash
cat ~/.dual-mem/runtime-token
```

不要把 `8765` 暴露到公网。安全组只放行 SSH（或你的隧道端口）。

## 每台电脑上

工具装在电脑上，记忆库在服务器上。一条命令连上：

```bash
dual-mem hub connect user@your-server      # 也可以写 ~/.ssh/config 里的 Host 名
```

它做三件事，安全模型不变：

1. 通过 SSH 读服务器上的 `~/.dual-mem/runtime-token` 和 `runtime.json`（Core 的端口）。token 不经过剪贴板，也不出现在命令行参数里。
2. 在本机挑一个空闲端口，建 `ssh -N -L 127.0.0.1:<本机端口>:127.0.0.1:<Core 端口>` 隧道，并把它装成登录服务（macOS LaunchAgent / Linux systemd --user），断线自动重连。本机已经有 MemHub 占着 `8765` 也不冲突。
3. 等隧道连通、用 token 请求一次 Core 确认身份，再把地址和 token 写进这台电脑的 `~/.dual-mem/hub.json`。

SSH 要能免密登录（key 或 agent）：隧道服务在后台运行，不能弹密码框。非默认端口或指定 key 用 `--ssh-arg` 透传，例如 `--ssh-arg=-p --ssh-arg=2222`；固定本机端口用 `--local-port 8766`。服务器上的 Hub 目录不是 `~/.dual-mem` 时用 `--remote-hub-dir`。

这台电脑上的 CLI、MCP、Hook 之后都走这条连接，不再使用本机那份记忆库。接入你实际在用的工具，然后把这台电脑上已有的会话送过去：

```bash
dual-mem hub install-host claude-code
dual-mem hub install-host codex

dual-mem hub scan --source claude-code
dual-mem hub scan --source codex
dual-mem hub recall "我之前做过哪些技术选择"
```

`scan` 在**这台电脑**上读转录，游标也留在这台电脑的 `~/.dual-mem/cursors.db`。记忆正文、Evidence 和向量只写在服务器的 `~/.dual-mem`。

不想连了：

```bash
dual-mem hub detach
```

`detach` 删掉 `hub.json` 并卸载 `connect` 建的隧道服务，这台电脑回到本机 Core 的自动发现。服务器上的库还在。服务器的 token 换了（例如删掉 `runtime-token` 后重启 Core），重新执行一次 `connect`。

### 不用 SSH 时

Tailscale、WireGuard 只负责把两台机器连上；MemHub 仍要落到本机 loopback，或者由你在服务器上做 TLS 反代。这时手动接：

```bash
dual-mem hub attach --base-url https://你的域名 --api-key "$(ssh user@your-server 'cat ~/.dual-mem/runtime-token')"
```

直接写 `http://公网IP:8765` 会被拒绝。

## 和本机模式的差别

| | 只在一台电脑上 | 服务器上的一个 Core |
|---|---|---|
| Core 在哪 | 这台电脑的 `127.0.0.1:8765` | 服务器的 `127.0.0.1:8765` |
| 要不要连接 | 不用，读 `runtime.json` | 每台电脑执行一次 `hub connect` |
| 工具配置 | 在这台电脑上 `install-host` | 也在每台电脑上 `install-host`，不要装到服务器上 |
| 记忆文件 | `~/.dual-mem` | 只在服务器的 `~/.dual-mem` |
| 断网 | 本机照常 | 隧道断了就写不了；不会在本地另存一份再合并 |
