# dual-mem 架构说明

dual-mem 采用「核心 SDK + 生态接入层」的分层设计。**SDK 是唯一的核心**，对外的
CLI / REST / MCP / Skill 都是围绕它的并列接入方式（frontends），互不依赖、可按需启用。

## 分层总览

```
        Agent / Cursor / Claude Desktop / 你的应用
                          │
        ┌─────────────┬───┴────────┬──────────────┐
        │             │            │              │
     Skill          MCP          REST            CLI          ← 生态接入层（并列）
   (使用说明)   (stdio/HTTP)   (FastAPI)    (dual-mem 命令)
        └─────────────┴───┬────────┴──────────────┘
                          │  都调用同一套 Python API
                ┌─────────▼─────────┐
                │   dual-mem SDK    │              ← 核心：MemoryClient
                │  (MemoryClient)   │
                └─────────┬─────────┘
                          │
        ┌─────────────────┼─────────────────────┐
        │                 │                      │
   Vector (Chroma)   Graph (Kuzu)        KV / 历史 (SQLite)   ← 存储层
```

Agent 侧推荐链路（与 Mem0 / Zep 一致的黄金架构）：

```
Agent / Cursor  ──MCP──▶  MCP Server（受 Skill 指导）  ──▶  dual-mem SDK  ──▶  Vector + KV
```

- **Skill** 不是代码层，而是给 agent 的「何时调用、如何解读结果」说明书，引导 agent 正确使用 MCP/CLI。
- **MCP** 是面向 agent 的标准协议接入，把 SDK 能力暴露为工具。
- **SDK** 是全部业务逻辑所在；REST/CLI/MCP 都只是薄封装。

## 各层职责

| 层 | 模块 | 职责 | 依赖 |
|---|---|---|---|
| 核心 SDK | `dual_mem.client.MemoryClient` | 写入/检索/演化/System2 全部逻辑，全 async | 仅依赖存储与 provider |
| CLI（SDK 前端） | `dual_mem.cli`（`dual-mem`） | SDK 的命令行外壳，`asyncio.run` 包装 | → SDK |
| REST | `dual_mem.api`（FastAPI） | HTTP 接口，鉴权 + 统一错误，契约见 `hy-api.md` | → SDK |
| MCP（生态） | `dual_mem.mcp`（`dual-mem-mcp`） | 面向 agent 的工具协议，支持 stdio / streamable-http | → SDK |
| Skill（生态） | `skills/dual-mem/SKILL.md` | 指导 agent 何时/如何调用 MCP/CLI | → MCP/CLI |
| 存储 | `dual_mem.storage` | Chroma 向量 / Kuzu 图 / SQLite 缓存与历史 | — |

> 关于目录结构：MCP/CLI 代码物理上仍位于 `dual_mem.*` 子包内（便于单包安装与
> 复用 `MemoryClient`），但**职责上与 SDK 并列**——它们只做协议转换，不含业务逻辑。
> 这与 Mem0 等项目把 `mem0` 核心与 `openmemory`(MCP) 同仓并存的取舍一致。

## 八层记忆框架（SDK 内部）

| 层 | 含义 | 产生方 |
|---|---|---|
| L0_BASIC_INFO | 结构化基础画像（姓名/年龄/所在地…） | System1 工具 |
| L1_RAW | 原始写入文本 | 写入即落 |
| L2_FACT | 抽取的离散事实 | System1 抽取 |
| L3_SUMMARY | 长文本摘要 | System1 摘要 |
| L4_IDENTITY | 身份/长期偏好 | System1 抽取 + 整理 |
| L5_KNOWLEDGE | 知识类记忆 | — |
| L6_SCHEMA | 跨证据归纳的行为模式 | System2 聚类 |
| L7_INTENTION | 用户的具体未来意图 | System2 |

- **System1（写侧，同步）**：Extractor → Reconciler（ADD/SUPERSEDE/DELETE，演化链）→ Summarizer。
- **System2（异步，仅 ultra）**：DBSCAN 聚类事实 → 出 L6 Schema / L7 Intention，写入图库；`digest()` 触发。
- **三路召回**：profile（画像）/ proactive（意图，仅 ultra）/ normal（事实知识），含演化链展开与 intent 加权 RRF。

## 三档模式

| 维度 | lite | pro | ultra |
|---|---|---|---|
| LLM 调用 | 无 | 写入时同步抽取/整理/摘要 | pro + System2 异步 |
| 写入层 | L1_RAW | L0–L4 | L0–L7 |
| System2 / 图库 | ✗ | ✗ | ✓ |
| proactive 召回 | 空 | 空 | 有 L7 意图 |

更多接入与部署细节见 [`mcp_integration.md`](./mcp_integration.md)。
