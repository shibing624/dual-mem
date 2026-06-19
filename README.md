[**🇨🇳中文**](https://github.com/shibing624/dual-mem/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/dual-mem/blob/main/README_EN.md)

# dual-mem: Dual-System Layered Memory SDK
[![PyPI version](https://badge.fury.io/py/dual-mem.svg)](https://badge.fury.io/py/dual-mem)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.11%2B-green.svg)](pyproject.toml)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/dual-mem.svg)](https://github.com/shibing624/dual-mem/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#社区与支持)

**dual-mem** 是面向 LLM 应用与 Agent 的**双系统分层记忆 SDK**。它把零散对话沉淀为结构化、可演化、可检索的长期记忆，并通过 **SDK / REST / MCP / CLI / Skill** 五种方式对外提供能力。

| 能力 | 说明 |
|------|------|
| **演化链** | 偏好/事实变更时显式 `supersedes` 指针；读侧自动展开历史（最新→最旧） |
| **System1 写路径** | Gate（LLM）→ Extract → fast-write L2/L4 → 异步 Reconcile 合并演化链 |
| **System2 沉淀** | dual 模式：DBSCAN 聚类 + 8 工具 ReAct → L6 Schema / L7 Intention |
| **三路召回** | profile / proactive / normal；默认 `hybrid` 读路径（无 LLM，有 embedding） |
| **多入口** | 五种 frontend 共享同一 `MemoryClient` 内核，extras 按需安装 |

## 🔥 News

- [2026/06/19] **v0.1.1**：Attentional Gate 改为 **LLM 主路径**（失败降级启发式）；依赖拆分为 `[api]` / `[cli]` / `[mcp]` extras；默认 `hybrid` 读路径。
- [2026/06/18] **v0.1.0**：首版开源 — system1 / dual 两档、演化链、REST `/v1/memories/` 契约、MCP 工具集。

## 架构

dual-mem 采用「**核心 SDK + 并列接入层**」设计：`MemoryClient` 承载全部业务逻辑，CLI / REST / MCP / Skill 只做协议转换。

```
        Agent / Cursor / 你的应用
              │
    ┌─────────┼─────────┬─────────┐
    │  Skill  │   MCP   │  REST   │  CLI     ← 接入层（extras 可选）
    └─────────┼─────────┴─────────┘
              ▼
       dual_mem.MemoryClient          ← 核心 SDK
              │
    Chroma (向量) + Kuzu (图) + SQLite (缓存/队列)
```

写侧与读侧概要：

```
写: Gate(1×LLM) → Extract(1×LLM) → fast-write L2/L4 → [async Reconcile / System2]
读: QueryUnderstanding → AnchorSearch(5路) → GraphExpander → FusionScorer → 三路分组
```

详细分层、八层记忆模型与模式对照见 [架构文档](docs/architecture.md) 与在线文档 [Architecture](https://shibing624.github.io/dual-mem/architecture/)。

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

```python
import asyncio
from dual_mem import MemoryClient


async def main():
    client = MemoryClient(mode="system1", storage_dir="./.dual_mem_data")

    await client.add(
        content="我最爱的编程语言是 Java，已经用了5年。",
        app_id="my_app",
        user_id="alice",
    )

    res = await client.search(
        query="用户的编程语言偏好",
        app_ids=["my_app"],
        user_id="alice",
    )
    for m in res.memories.profile:
        print(m.content)

    await client.aclose()


asyncio.run(main())
```

配置写入 `~/.dual_mem/config.yaml`（参考 `config.example.yaml`）。**LLM 与 Embedding 两套 API key 均必填**。

```yaml
mode: system1
llm_api_key: sk-...
embed_api_key: sk-...
embed_model: text-embedding-3-small
embed_dim: 1536
```

## 功能特性

- **Async-First** — `MemoryClient` 全异步；CLI/REST 内部 `asyncio.run` 包装
- **Attentional Gate** — LLM 三维度打分（novelty / relevance / arousal），失败降级启发式
- **Fast-write + 异步 Reconcile** — 写路径低延迟；`reconcile_sync=true` 可切强一致
- **演化链** — 层内 `supersedes` 指针，软删保留历史；读侧 `evolution_chain` 展开
- **Hybrid 读路径（默认）** — 5 路 anchor 并行 + 图 1-hop + FusionScorer；中文时间词解析
- **System2 ReAct（dual）** — 8 工具 function-calling 循环，产出 L6/L7 与图边
- **OpenAI 兼容** — LLM / Embed 任意兼容端点（OpenAI、DashScope、智谱、本地 vLLM 等）
- **多租户隔离** — `app_id` + `user_id` + 可选 `agent_id` / `session_id`

## 两档模式

| 维度 | system1（默认） | dual |
|---|---|---|
| 每次 `add` LLM（fast-write） | ~2（Gate + Extract；长文本 +1 summarize） | 同 system1 + 异步 1~N |
| 写入层 | L0 / L1(SHADOW) / L2 / L3 / L4 | + L5 / L6 / L7 |
| 图库（Kuzu） | 关 | 开 |
| System2 | 否 | ReconcileWorker + ReAct Agent + Sweeper |

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

REST 契约：`POST /v1/memories/`、`POST /v1/memories/search`、`GET|DELETE /v1/memories/{id}` 等。详见 [MCP 接入](docs/mcp_integration.md)。

## 示例

查看 [examples/](https://github.com/shibing624/dual-mem/tree/main/examples) 获取完整示例，涵盖：

| 类别 | 内容 |
|------|------|
| **SDK 基础** | system1 写入、演化链、多轮 `messages` 输入 |
| **System2** | dual 模式 `digest`、ReAct 蒸馏、跨域 Schema |
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
- [mem0ai/mem0](https://github.com/mem0ai/mem0) — 分层记忆 SDK 生态的先行探索
