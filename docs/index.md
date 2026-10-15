# dual-mem

**dual-mem** 是面向 LLM 应用与 Agent 的分层记忆 SDK。`system1` 和 `dual` 共用同一条写入：保留 `L0_RAW` → 生成候选 → JEV 批量裁决 → 追加版本并链接历史。`dual` 只在显式 `digest()` 时再生成 `L3_DERIVED`。读侧是 hybrid 混合召回，默认只返回当前版本。

## 核心能力

| 能力 | 说明 |
|------|------|
| **四层** | `L0_RAW` 原文、`L1_FACT` 事实/事件/意图、`L2_PROFILE` 画像、`L3_DERIVED` 跨证据推导 |
| **JEV 裁决** | 已知与更新分开。动作是 `ADD_*` / `SKIP` / `DELETE` / `RAW_ONLY`；关系是 `OVERRIDE` / `MERGE` / `SUPPLEMENT` / `TEMPORAL` / `NEGATE` / `CONFLICT` |
| **演化链** | `supersedes` / `superseded_by`。`update` 返回新 id，`delete` 归档，历史用 `is_latest=false` 保留 |
| **三路召回** | profile / proactive / normal。默认检索不返回 `L0_RAW`；核对原话用 `search_conversation` |
| **多入口** | SDK / REST / MCP / CLI / Skill 共享 `MemoryClient` |

## 快速链接

- [安装与配置](getting-started/installation.md)
- [5 分钟上手](getting-started/quickstart.md)
- [架构说明](architecture.md)
- [四层记忆](memory_layers.md)
- [JEV 决策与迁移](jev-memory-design.md)
- [MCP 接入](mcp_integration.md)
- [示例代码](https://github.com/shibing624/dual-mem/tree/main/examples)

## 安装

```bash
pip install dual-mem
pip install dual-mem[all]   # + REST / CLI / MCP
```

详见 [Installation](getting-started/installation.md)。
