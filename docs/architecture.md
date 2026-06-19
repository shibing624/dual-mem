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
   (使用说明)  (本地/云端*)    (FastAPI)    (dual-mem 命令)
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
- **MCP** 是面向 agent 的标准协议接入，把 SDK 能力暴露为工具。*本地 uvx MCP 已实现；云端 HTTP MCP 规划走 REST 同一契约，见 [`mcp_integration.md`](./mcp_integration.md)。
- **SDK** 是全部业务逻辑所在；REST/CLI/MCP 都只是薄封装。

## 各层职责

| 层 | 模块 | 职责 | 依赖 |
|---|---|---|---|
| 核心 SDK | `dual_mem.client.MemoryClient` | 写入/检索/演化/System2 全部逻辑，全 async | 仅依赖存储与 provider |
| CLI（SDK 前端） | `dual_mem.cli`（`dual-mem`） | SDK 的命令行外壳，`asyncio.run` 包装 | → SDK |
| REST | `dual_mem.api`（FastAPI） | HTTP 接口，鉴权 + 统一错误，契约见 `docs/architecture.md` | → SDK |
| MCP（生态） | `dual_mem.mcp`（`dual-mem-mcp`） | 面向 agent 的工具协议；**本地** stdio / streamable-http 已实现；**云端** HTTP MCP 规划走 REST 同一契约 | → SDK（云端经 REST → SDK） |
| Skill（生态） | `skills/dual-mem/SKILL.md` | 指导 agent 何时/如何调用 MCP/CLI | → MCP/CLI |
| 存储 | `dual_mem.storage` | Chroma 向量 / Kuzu 图 / SQLite 缓存与历史 | — |

> 关于目录结构：MCP/CLI 代码物理上仍位于 `dual_mem.*` 子包内（便于单包安装与
> 复用 `MemoryClient`），但**职责上与 SDK 并列**——它们只做协议转换，不含业务逻辑。
> 这与 Mem0 等项目把 `mem0` 核心与 `openmemory`(MCP) 同仓并存的取舍一致。

## 八层记忆框架（SDK 内部）

| 层 | 含义 | 产生方 |
|---|---|---|
| L0_BASIC_INFO | 结构化基础画像（姓名/年龄/所在地…） | System1 工具 |
| L1_RAW | 原始写入文本（system1/dual 写后转 SHADOW） | 写入即落 |
| L2_FACT | 抽取的离散事实 | System1 抽取 |
| L3_SUMMARY | 长文本摘要（≥500 字才生成） | System1 摘要 |
| L4_IDENTITY | 身份/长期偏好 | System1 抽取 + 整理 |
| L5_KNOWLEDGE | 知识类记忆 | — |
| L6_SCHEMA | 跨证据归纳的行为模式 | System2 聚类 |
| L7_INTENTION | 用户的具体未来意图 | System2 |

- **System1（写侧）**：Attentional Gate（`agent/gate.py`，LLM 主路径 + 启发式降级）→ Extractor（1 次 LLM，出 identity/facts/intentions/emotion/basic_info）→ fast-write 直接落 L2/L4 → 入队 reconcile 任务 → Summarizer（仅长文本出 L3）。默认 **fast-write + 异步 reconcile**（`reconcile_sync=false`，最终一致）；置 `reconcile_sync=true` 可在写侧同步跑 Reconciler（强一致）。
- **System2（异步，仅 dual）**：`ReconcilerWorker` 排空 reconcile 队列（ADD/SUPERSEDE/DELETE，构建演化链、软删 fast-write 原件）→ `System2Agent` 两阶段 DBSCAN 聚类后跑**真 ReAct 循环**（`chat_with_tools` + `s2_tools` 八工具，`tool_choice="auto"`，至多 `system2_max_iters=10` 轮）写 L6 Schema / L7 Intention / 图边 → `CrossDomainSweeper`（受 `cross_domain_enable` 控制，默认关）将 ≥5 条基础 Schema 升维为核心 Schema。触发由 `system2_trigger_mode` ∈ `{per_write, manual, scheduled}` 决定，同 user 用 `asyncio.Lock` 串行；`manual` 需 `client.digest()`。
- **三路召回（读侧，`reader_mode=hybrid` 默认）**：**不调用 LLM**（意图分类、时间词解析均为 regex/关键词启发式），但会 **调用 Embedding API** 做 query 向量化，并在 Chroma/Kuzu 上做向量检索。管线：`query_understanding` → 五路并行 `anchor_search`（语义 + 关键词 + 图）→ `graph_expander` 1-hop（dual 有图时）→ `fusion_scorer`（W(d) 时效 + access_count + 多路 RRF）→ 按 profile / proactive / normal 分组 → 演化链展开。读后 fire-and-forget `ReconsolidationHook` 累加访问计数并建弱关联边。
- **legacy 读路径（`reader_mode=legacy`）**：旧版三路向量召回 + normal 路 BM25+RRF 重排；同样无 LLM，有 embedding。

## 两档模式

| 维度 | system1（默认） | dual |
|---|---|---|
| 写侧 LLM 调用 | Gate + Extract（长文本 +1 summarize） | 同 system1；reconcile + S2 ReAct 异步 |
| 写入层 | L0–L4 | L0–L7 |
| System2 / 图库 | ✗ | ✓ |
| proactive 召回 | 空 | 有 L7 意图 |

> 已知缺口（勿当作已实现）：hybrid 读路径暂未用 `query_understanding.target_layers` 路由（写死层列表）；读侧 `ReconsolidationHook` 会入队 `reconsolidation` 任务，但 System2 仅记 `RECONSOLIDATION_DRAIN` 日志后出队、无专用 ReAct，故该再巩固沉淀目前等于 no-op；`sdk_models.ReadResult`（trace 字段）已定义但 `Reader.search` 仅返回 `SearchMemories`。

更多接入与部署细节见 [`mcp_integration.md`](./mcp_integration.md)。
