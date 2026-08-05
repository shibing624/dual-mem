[**🇨🇳中文**](https://github.com/shibing624/dual-mem/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/dual-mem/blob/main/README_EN.md)

# dual-mem: Dual-System Layered Memory SDK
[![PyPI version](https://badge.fury.io/py/dual-mem.svg)](https://badge.fury.io/py/dual-mem)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.11%2B-green.svg)](pyproject.toml)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/dual-mem.svg)](https://github.com/shibing624/dual-mem/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#社区与支持)

**dual-mem** 让 Agent **真正记住用户**——不是堆聊天记录，而是像人一样：**先快速记下今天说了什么，再慢慢整理成用户画像、习惯和未来打算**。分层存储、偏好会变就更新、检索时自动带出相关记忆。通过 **SDK / REST / MCP / CLI / Skill** 即插即用。

## 记忆架构（L0–L7）

dual-mem 实现 **七个记忆层（L0–L4、L6、L7）**——借鉴人脑「先感知、再编码、再巩固、再抽象」的分层过程，从原始对话一路沉淀到画像、习惯与打算，并带 **演化链**（偏好变了留历史，不粗暴覆盖）：

| 层 | 存什么 | 举例 |
|---|---|---|
| **L0 基础画像** | 结构化档案 | 姓名、年龄、城市 |
| **L1 原始对话** | 完整交互记录 | 多轮 chat 原文（提炼后归档） |
| **L2 事实** | 可核对的具体信息 | 「下周去北京出差」 |
| **L3 摘要** | 长文压缩 | 一篇长笔记的要点 |
| **L4 身份与偏好** | 稳定的「你是谁、喜欢什么」 | 「主力语言是 Python」 |
| **L6 行为模式** | 跨多条事实归纳的习惯 | 「做事前爱列清单、爱分类整理」 |
| **L7 意图** | 未来计划与目标 | 「准备跑马拉松」 |

偏好变了（Java → Python）不会覆盖旧记录，而是 **演化链** 保留历史，读的时候给你 **最新版 + 变更轨迹**。

## 双系统：短期记忆 × 深度记忆

| | **System1 · 短期记忆** | **System2 · 深度记忆** |
|---|---|---|
| **像什么** | 当场听懂并记下来 | 睡后整理、越聊越懂你 |
| **什么时候** | 每次 `add` 立刻完成 | `dual` 模式显式调用 `digest()` |
| **产出** | L0–L4：事实、偏好、画像 | L6 行为模式、L7 意图、知识关联 |
| **怎么用** | `mode="system1"`（默认） | `mode="dual"` |

```
对话 add ──▶ System1 短期记忆（秒级可用）
                │
                └──▶ System2 深度记忆（dual：digest 时巩固）
```

检索侧按 **画像 / 主动推荐 / 常规记忆** 三路分组，语义 + 关键词混合召回，**读记忆不额外调 LLM**（只需 embedding）。

## 为什么选 dual-mem

| | |
|---|---|
| **开箱即用** | `SyncMemoryClient` 三行写入 + 搜索；配置首次启动自动生成 |
| **Agent 友好** | 多轮 `messages` 一次写入；与 Agentica 等框架对接见 [Quickstart](docs/getting-started/quickstart.md) |
| **会进化** | 偏好更新走演化链，不是粗暴覆盖 |
| **五种接入** | 同一套 `MemoryClient`，按需装 REST / MCP / CLI |

## 🔥 News

- [2026/06/20] **v0.1.2**：MemoryOperations 统一 REST/MCP；SyncMemoryClient；CLI `messages`；post-extract embed 合并；MCP 启动修复与文档。
- [2026/06/19] **v0.1.1**：依赖拆分为 `[api]` / `[cli]` / `[mcp]` extras；默认 `hybrid` 读路径。
- [2026/06/18] **v0.1.0**：首版开源 — system1 / dual 两档、演化链、REST `/v1/memories/` 契约、MCP 工具集。

## 架构一览

```
  你的 Agent / 应用
        │  SDK · REST · MCP · CLI · Skill
        ▼
   MemoryClient（写入 System1 / 检索 / 演化链）
        │
   向量 + 图 + 本地存储
```

实现细节（Extractor 提交判定、Reconcile、混合检索管线等）见 [架构文档](docs/architecture.md) · [在线文档](https://shibing624.github.io/dual-mem/architecture/)。

## 安装

需要 Python ≥ 3.11。

```bash
pip install dual-mem              # 核心 SDK
pip install dual-mem[all]         # + REST / CLI / MCP
```

开发安装：

```bash
git clone https://github.com/shibing624/dual-mem.git
cd dual-mem
pip install -e ".[dev]"
```

| Extra | 依赖 | 用途 |
|---|---|---|
| `api` | fastapi, uvicorn | `dual-mem serve`、REST API |
| `cli` | typer | `dual-mem add/search/...` |
| `mcp` | mcp | `dual-mem-mcp`、Cursor MCP |
| `all` | 以上全部 | 五种入口一次装齐 |

## 快速开始

脚本 / 同步应用（推荐入门）：

```python
from dual_mem import SyncMemoryClient

with SyncMemoryClient(mode="system1", storage_dir="./.dual_mem_data") as client:
    client.add(
        content="我最爱的编程语言是 Java，已经用了5年。",
        user_id="alice",
    )
    res = client.search(
        query="用户的编程语言偏好",
        user_id="alice",
    )
    for m in res.memories.profile:
        print(m.content)
```

FastAPI / asyncio Agent 用异步客户端（同一内核，可 `await`）：

```python
import asyncio
from dual_mem import MemoryClient


async def main():
    client = MemoryClient(mode="system1", storage_dir="./.dual_mem_data")
    await client.add(content="...", user_id="alice")
    await client.aclose()


asyncio.run(main())
```

配置位于 `~/.dual_mem/config.yaml`（**首次启动自动创建**）。**LLM 与 Embedding 两套 API key 均必填**。

→ 参数详解、多轮对话、`aclose()`、Agent 集成最佳实践见 **[Quickstart 文档](docs/getting-started/quickstart.md)**（在线：[Quickstart](https://shibing624.github.io/dual-mem/getting-started/quickstart/)）。

## 功能特性

- **七个实现层 + 演化链** — 从 raw 对话到行为模式、意图，偏好变更可追溯
- **System1 短期记忆** — 写入即结构化；由 Extractor 决定是否提交衍生记忆
- **System2 深度记忆**（dual）— 在明确业务边界调用 `digest()` 归纳 L6/L7
- **混合检索** — 语义 + 关键词；画像 / 主动 / 常规三路召回
- **同步 & 异步 SDK** — `SyncMemoryClient` 给脚本；`MemoryClient` 给 FastAPI / Agent
- **OpenAI 兼容** — 任意 LLM / Embedding 端点
- **单产品默认** — 省略 `app_id` 时用 `default_app_id`（默认 `"default"`）；`user_id` 为主隔离键

## 两档模式

| | **system1**（默认） | **dual** |
|---|---|---|
| 短期记忆 | ✓ | ✓ |
| 深度记忆 | — | ✓（显式 `digest()` 巩固） |
| 适合 | 个人助手、客服、快速接入 | 需要行为画像、意图推断的深度场景 |
| 图关联 | 关 | 开（记忆之间可关联） |

## REST / MCP / CLI

```bash
pip install dual-mem[all]

dual-mem serve --host 0.0.0.0 --port 8000          # REST
dual-mem-mcp                                          # MCP stdio
dual-mem-mcp --transport streamable-http --port 8765 # MCP HTTP /mcp
dual-mem add --content "用户喜欢咖啡" --app-id default --user-id u1
dual-mem search "饮品偏好" --app-id default --user-id u1
dual-mem digest   # dual：触发 System2
```

REST 契约：`POST /v1/memories/`、`POST /v1/memories/search`、`GET|DELETE /v1/memories/{id}` 等。MCP 分 **本地 uvx**（已实现）与 **云端 HTTP**（REST 底座已就绪，HTTP MCP 待封装）两条路径，详见 [MCP 接入](docs/mcp_integration.md)。

## 示例

查看 [examples/](https://github.com/shibing624/dual-mem/tree/main/examples) 获取完整示例，涵盖：

| 类别 | 内容 |
|------|------|
| **SDK 基础** | 短期记忆写入、演化链、多轮 `messages` |
| **深度记忆** | dual 模式、显式 `digest()` 巩固 |
| **REST** | FastAPI TestClient 走 `/v1/memories/` 全契约 |
| **CLI** | `dual-mem` 子命令 add / search / list / delete |

[→ 查看完整示例目录](https://github.com/shibing624/dual-mem/blob/main/examples/README.md)

```bash
python examples/01_system1.py
python examples/02_dual.py
python examples/03_rest_api.py
python examples/04_cli.py
```

> 示例需真实 LLM + Embedding API key，会产生少量费用。

## 文档

完整使用文档：**https://shibing624.github.io/dual-mem**

本地预览：

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

## 社区与支持

- **GitHub Issues** — [提交 issue](https://github.com/shibing624/dual-mem/issues)
- **微信群** — 添加微信号 `xuming624`，备注 "llm"，加入技术交流群

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" />

## 引用

如果您在研究中使用了 dual-mem，请引用：

> Xu, M. (2026). dual-mem: Dual-System Layered Memory SDK for LLM Applications. GitHub. https://github.com/shibing624/dual-mem

## 许可证

[Apache License 2.0](LICENSE)

## 贡献

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 致谢

- [chromadb/chroma](https://github.com/chroma-core/chroma) — 向量存储
- [kuzudb/kuzu](https://github.com/kuzudb/kuzu) — 图存储
- [tiangolo/fastapi](https://github.com/tiangolo/fastapi) — REST 层
- [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) — MCP 集成
- [mem0ai/mem0](https://github.com/mem0ai/mem0) — 记忆 SDK 生态
