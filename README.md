[**🇨🇳中文**](https://github.com/shibing624/dual-mem/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/dual-mem/blob/main/README_EN.md)

<div align="center">

# dual-mem

**All your agents. One Memory.**

让 AI 真正记住你——偏好变了它知道，换个工具它也知道。

[![PyPI version](https://badge.fury.io/py/dual-mem.svg)](https://badge.fury.io/py/dual-mem)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.11%2B-green.svg)](pyproject.toml)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/dual-mem.svg)](https://github.com/shibing624/dual-mem/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#社区与支持)

</div>

---

你已经告诉 AI 三次"我现在不吃辣了"，它还在推荐川菜馆。
在 Cursor 里交代过的代码规范，Codex 并不知道；在 Claude Code 里定下的技术选型，换到 Hermes 又要从头讲一遍。

**dual-mem 解决这两件事：**

- 🧠 **记得准**：不是堆聊天记录，而是像人一样整理——事实、画像、习惯分开存；你改了主意，它更新而不是混淆。
- 🔗 **记得通**：MemHub 让你常用的 AI 工具共享同一份个人记忆。记忆是你的长期资产，不被任何单一工具锁定——每次开口，都从懂你开始。

## 效果：同场对比，全面领先

同一个模型（Qwen3.8-27B）负责抽取、作答与评分，同一套 embedding、同一批题目——**只换记忆系统**：

| 记忆系统 | LoCoMo<br>(308 题) | LongMemEval<br>(24 题) | PersonaMem-v2<br>(476 题) | 写入耗时<br>(秒/段对话) |
|---|:---:|:---:|:---:|:---:|
| **dual-mem** | **74.7** | **91.7** | **50.0** | **503** |
| hy-memory 2.0.4 | 66.6 | 70.8 | 43.5 | 2485 |
| mem0 | 65.6 | 83.3 | 47.9 | 808 |

> 写入耗时为 LoCoMo 每段长对话的中位数。评测协议、显著性检验、题型细分和复现命令见 [评测说明](docs/benchmarks.md)。

## 三分钟上手

```bash
pip install dual-mem
```

首次运行会生成 `~/.dual_mem/config.yaml`，填入模型服务的 API key 即可（任意 OpenAI 兼容端点）。

```python
from dual_mem import SyncMemoryClient

with SyncMemoryClient() as memory:
    memory.add(content="我住在上海，最近开始不吃辣了。", user_id="alice")
    memory.add(content="下个月我要搬去北京工作。", user_id="alice")

    res = memory.search(query="给 alice 推荐一家餐厅", user_id="alice")
    for m in res.memories.profile + res.memories.normal:
        print(m.content)
```

它会记住 alice 不吃辣、即将搬去北京——而且知道"上海"已经是过去时。

想在自己的 Agent 里用？异步客户端、多轮对话写入和 Agent 集成写法见 [快速开始](docs/getting-started/quickstart.md)。

## MemHub：你的 AI 工具共享一份记忆

```text
   Claude Code   Codex   Cursor   Hermes   Agentica
        \          |        |        |       /
         ─────────────  MemHub  ─────────────
                         │
                你的个人记忆（本机 / 你的云主机）
```

```bash
pip install "dual-mem[all]"

dual-mem hub install-service       # 后台常驻，开机自启
dual-mem hub install-host all      # 一键接入本机已安装的 AI 工具
dual-mem hub scan --source all     # 把过去的对话也变成记忆

dual-mem hub recall "我之前做过哪些技术选择"
```

- **零配置**：工具自动发现本机记忆服务，不用填地址和密钥。
- **不打断你**：对话先存下来立刻返回，整理在后台进行，不拖慢任何工具。
- **历史不浪费**：已有的会话记录一键导入，第一天就"懂你"。
- **数据归你**：默认只存在你的电脑上；也可以放到你自己的云主机，多台电脑连同一份记忆。

当前已支持 **Claude Code、Codex、Cursor、Hermes、Agentica**，更多工具见 [Roadmap](#roadmap)。详见 [MemHub 文档](docs/memhub.md)。

## 适合谁用

| 你是… | dual-mem 帮你… |
|---|---|
| **重度 AI 工具用户** | 用 MemHub 让各个 AI 工具共享你的偏好、规范和项目背景，不再重复交代 |
| **Agent / 应用开发者** | 三行代码给产品加上长期记忆；SDK、REST、MCP、CLI 任选接入 |
| **个人助手 / 客服产品** | 记住每位用户的画像与变化，回答越用越贴心 |
| **研究者** | 同场对比 mem0、hy-memory 的评测工具开箱即用 |

## 为什么记得更准

- **偏好变了会更新，而不是混淆**：每条记忆都有版本，"住上海 → 搬北京"是一次更新，旧版本留作历史，回答默认用最新的。
- **事实与画像分开整理**：说过的话、明确的事实、你是什么样的人，各存一层，按需取用。
- **专用决策模型把关**：写入时由一个轻量决策模型判断"新信息、重复，还是更新"，比让大模型自己判断更稳、更便宜。
- **原话永远可查**：整理过的记忆之外，原始对话始终保留，随时可以核对出处。

想了解实现原理？见 [开发者文档](#开发者文档)。

## 接入方式

| 方式 | 适合 | 一句话 |
|---|---|---|
| **Python SDK** | 自己写 Agent / 应用 | `pip install dual-mem` |
| **MemHub** | 日常使用多个 AI 工具 | `dual-mem hub install-host all` |
| **MCP** | Cursor、Claude Desktop 等支持 MCP 的工具 | `dual-mem-mcp` |
| **REST API** | 其他语言或独立服务 | `dual-mem serve` |
| **CLI** | 脚本、调试 | `dual-mem add` / `dual-mem search` |
| **框架插件** | Agentica、Claude Code Hook、Hermes、OpenClaw | 见 [集成文档](docs/integrations.md) |

## Roadmap

- [x] 分层记忆 + 版本演化：偏好更新可追溯
- [x] 专用决策模型把关写入
- [x] MemHub：5 个 AI 工具共享记忆、历史会话导入、后台整理
- [x] 云主机单实例：多台电脑连同一份记忆
- [x] LoCoMo / LongMemEval / PersonaMem-v2 同场评测领先
- [ ] MemHub 接入更多工具：OpenCode、WorkBuddy、Gemini CLI、Windsurf、Trae 等，目标 10+ 种
- [ ] 云端 HTTP MCP：远程 Agent 直接连接托管记忆
- [ ] 多跳推理类问题的检索增强（LoCoMo multi-hop 仍有提升空间）
- [ ] 多设备记忆同步：离线可用、联网自动合并
- [ ] 记忆可视化管理：查看、编辑、删除你的记忆
- [ ] 隐私分级：敏感记忆只留本机
- [ ] 补充更多公开评测（CodeMemoryBench 等，待数据公开）

欢迎在 [Issues](https://github.com/shibing624/dual-mem/issues) 投票或提需求。

## 🔥 News

- [2026/10/23] LoCoMo 同场评测：dual-mem 74.7，领先 hy-memory 与 mem0 8 分以上。
- [2026/10/22] 精简配置：去掉评测无收益的开关，默认即最佳设置。
- [2026/10/15] **v0.1.3**：新一代分层记忆与决策模型写入（存储格式不兼容旧版，需新目录）。
- [2026/06/18] **v0.1.0**：首版开源。

完整变更见 [CHANGELOG](CHANGELOG.md)。

## 开发者文档

<details>
<summary>展开：架构、配置与原理（可选阅读）</summary>

| 文档 | 内容 |
|---|---|
| [安装与配置](docs/getting-started/installation.md) | extras、配置文件、凭证与优先级 |
| [快速开始](docs/getting-started/quickstart.md) | 同步 / 异步客户端、多轮写入、Agent 集成 |
| [架构说明](docs/architecture.md) | SDK 与各接入层的关系、存储 |
| [四层记忆](docs/memory_layers.md) | L0 原文 / L1 事实 / L2 画像 / L3 推导 |
| [写入决策设计](docs/jev-memory-design.md) | 决策模型动作与关系、版本链、并发 |
| [MemHub](docs/memhub.md) | Core 服务、宿主接入、云端单实例 |
| [MCP / REST](docs/mcp_integration.md) | 工具 ↔ HTTP ↔ SDK 对照 |
| [生态集成](docs/integrations.md) | Agentica / Claude Code / Hermes / OpenClaw |
| [评测说明](docs/benchmarks.md) | 协议、题集、复现与结果细分 |
| [示例](examples/README.md) | 可运行的完整示例 |

在线文档：**https://shibing624.github.io/dual-mem** · 本地预览：`pip install -r requirements-docs.txt && mkdocs serve`

</details>

## 社区与支持

- **GitHub Issues** — [提交 issue](https://github.com/shibing624/dual-mem/issues)
- **微信群** — 添加微信号 `xuming624`，备注 "llm"，加入技术交流群

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" />

## 引用

如果您在研究中使用了 dual-mem，请引用：

> Xu, M. (2026). dual-mem: Dual-System Layered Memory SDK for LLM Applications. GitHub. https://github.com/shibing624/dual-mem

## 许可证 · 贡献 · 致谢

[Apache License 2.0](LICENSE) · 欢迎贡献，见 [CONTRIBUTING.md](CONTRIBUTING.md)

感谢 [Chroma](https://github.com/chroma-core/chroma)、[Kuzu](https://github.com/kuzudb/kuzu)、[FastAPI](https://github.com/tiangolo/fastapi)、[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)、[mem0](https://github.com/mem0ai/mem0)。
