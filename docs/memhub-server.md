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

工具装在电脑上，记忆库在服务器上。电脑先把服务器的 loopback 接到自己的 loopback。

本地如果已经有 MemHub 占着 `8765`，隧道换一个本机端口，例如 `8766`。下面按这个写。

```bash
ssh -N -L 8766:127.0.0.1:8765 user@your-server
```

这条命令的意思是：

- `-N`：只转发端口，不打开远程 shell。
- `-L 8766:127.0.0.1:8765`：本机访问 `127.0.0.1:8766` 时，SSH 把流量送到**服务器自己的** `127.0.0.1:8765`。
- 加密在 SSH 里完成。MemHub 看到的仍是 loopback，所以可以用 `http://127.0.0.1`。

另开一个终端，用服务器上那把 token 接上。`attach` 会先请求一次 Core，成功后把地址和 key 写进这台电脑的 `~/.dual-mem/hub.json`：

```bash
dual-mem hub attach \
  --base-url http://127.0.0.1:8766 \
  --api-key "$(ssh user@your-server 'cat ~/.dual-mem/runtime-token')"
```

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

`detach` 删掉 `hub.json`，这台电脑回到本机 Core 的自动发现。服务器上的库还在。

隧道要一直开着。SSH 断了，`attach` 里的地址还在，但请求会失败。可以用 `autossh` 或系统服务把上面的 `ssh -N -L` 挂住。Tailscale、WireGuard 只负责把两台机器连上；MemHub 仍要通过隧道落到本机 loopback，或者由你在服务器上做 TLS 反代之后，用 `https://你的域名` 执行 `attach`。直接写 `http://公网IP:8765` 会被拒绝。

## 和本机模式的差别

| | 只在一台电脑上 | 服务器上的一个 Core |
|---|---|---|
| Core 在哪 | 这台电脑的 `127.0.0.1:8765` | 服务器的 `127.0.0.1:8765` |
| 要不要 `attach` | 不用，读 `runtime.json` | 要。每台电脑执行一次 |
| 工具配置 | 在这台电脑上 `install-host` | 也在每台电脑上 `install-host`，不要装到服务器上 |
| 记忆文件 | `~/.dual-mem` | 只在服务器的 `~/.dual-mem` |
| 断网 | 本机照常 | 隧道断了就写不了；不会在本地另存一份再合并 |
