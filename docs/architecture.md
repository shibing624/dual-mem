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

## 四个处理阶段

层定义见 [memory_layers.md](memory_layers.md)，调用契约与迁移见
[jev-memory-design.md](jev-memory-design.md)，可交互流程见
[jev-memory-flow.html](design/jev-memory-flow.html)。

| 层 | 含义 | 产生方 |
| --- | --- | --- |
| L0_RAW | 原始来源证据 | 写入即保留 |
| L1_FACT | 明确事实、独立事件、明确意图 | 生成候选 → JEV 裁决 |
| L2_PROFILE | 身份、画像、状态维度 | JEV representation |
| L3_DERIVED | 跨证据模式、摘要、推导意图 | 显式 digest → 生成 + JEV |

System1 两种模式共用：保留来源 → 一次生成候选 → 批量 JEV action/target/representation
→ 批量 embedding → 校验 scope epoch/revision → durable append/link。
MERGE 才增加文本合并及无损校验。默认 latest 读取，历史可显式查看。

dual 写入后登记 scope；digest 按证据数和字符预算分页，准确确认 processed_ids，
生成与 JEV 校验后提交 L3。不同 scope 默认并行 4，同 scope 有序处理。失败任务保留 pending。
版本提交与图投影之间通过持久待处理记录恢复，图只作为证据关系投影。

混合检索调用 Embedding，不调用语义决策模型。画像与事实只召回 latest；
图结果须对照 VDB 最新状态。关键词保留在读侧重排，与写入判断无关。

| 维度 | system1 | dual |
| --- | --- | --- |
| 写入 | 同一生成 + JEV 管线 | 同左 |
| 产物 | L0 / L1 / L2 | 同左，digest 增加 L3 |
| System2 / 图库 | 关闭 | 显式 digest |
| 版本保留 | 全部保留 | 全部保留 |

更多接入与部署细节见 [mcp_integration.md](mcp_integration.md)。
