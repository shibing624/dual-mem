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

## 七个实现层（SDK 内部）

各层语义、产生方、读路径归属的权威说明见 [`memory_layers.md`](./memory_layers.md)。

| 层 | 含义（对标 hy_memory） | 产生方 |
|---|---|---|
| L0_BASIC_INFO | 基础信息层：结构化基础画像（姓名/年龄/所在地…） | System1 `BasicProfile` |
| L1_RAW | 原始对话层：写入即落、Append-Only | 写入即落 |
| L2_FACT | 原子事实层：抽取的离散、版本化事实 | System1 抽取 |
| L3_SUMMARY | 会话摘要层：超过配置阈值的长文本摘要（默认关闭） | System1 摘要 |
| L4_IDENTITY | 身份画像层：身份/长期偏好 | System1 抽取 + 整理 |
| L6_SCHEMA | 心智模型层：跨证据归纳的行为模式 | System2 聚类 |
| L7_INTENTION | 前瞻意图层：用户的具体未来意图 | System2 |

- **System1（写侧）**：L1_RAW 先落盘 → Extractor（1 次 LLM，输出 identity/facts/intentions/basic_info 与 `is_ephemeral`）→ 提交判定 → fast-write 直接落 L0/L2/L4。摘要默认关闭；开启后长文本可写 L3。System1 不产生无人消费的 reconcile 队列；`reconcile_sync=true` 是显式强一致选项。
- **System2（仅 dual）**：写入登记待处理 scope；只有 `await client.digest()` 才依次运行 `ReconcilerWorker` 与 `System2Agent`。后者对新鲜 L2/L4 做 DBSCAN 聚类，再通过 single-shot 或 ReAct 工具调用写 L6 Schema、L7 Intention 与图边。SDK 不启动 per-write 或 scheduled 后台任务。
- **混合召回（读侧）**：不调用 LLM，但会调用 Embedding API。管线并行召回语义 VDB、L0 profile 和 L6 graph schema，在语义召回池内做 BM25 重排，融合图证据与演化链后按 profile / proactive / normal 分组。时间窗与派生层分别由 `created_after`、`include_derived` 显式控制，不从 query 猜测意图。

## 两档模式

| 维度 | system1（默认） | dual |
|---|---|---|
| 写侧 LLM 调用 | Extract（显式启用后长文本 +1 summarize） | 同 system1；digest 时 reconcile + S2 |
| 写入层 | L0–L4 | L0–L4、L6–L7 |
| System2 / 图库 | ✗ | ✓ |
| proactive 召回 | 空 | 有 L7 意图 |

> 读路径说明：关键词通道只重排语义召回池，不做全库扫描。融合权重与证据加成由 `hybrid_w_sem`、`hybrid_w_bm25`、`hybrid_evidence_boost_max`、`hybrid_evidence_saturate` 控制；`client.search(debug=True)` 返回精简的 `ReadResult`。

更多接入与部署细节见 [`mcp_integration.md`](./mcp_integration.md)。
