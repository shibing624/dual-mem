# dual-mem

**dual-mem** 是面向 LLM 应用与 Agent 的**双系统分层记忆 SDK**：写侧 System1 同步认知 + System2 异步沉淀，读侧 hybrid 混合召回，演化链显式追踪偏好与事实变更。

## 核心能力

| 能力 | 说明 |
|------|------|
| **演化链** | 写侧 `supersedes` / `superseded_by` 双向指针；读侧自动展开历史版本 |
| **System1 写路径** | Attentional Gate（LLM）→ Extract → fast-write L2/L4 → 异步 Reconcile |
| **System2 沉淀（dual）** | DBSCAN 聚类 + 8 工具 ReAct Agent → L6 Schema / L7 Intention |
| **三路召回（零 LLM）** | profile / proactive / normal；默认 hybrid anchor + fusion |
| **多入口** | SDK / REST / MCP / CLI / Skill 共享 `MemoryClient` |

## 快速链接

- [安装与配置](getting-started/installation.md)
- [5 分钟上手](getting-started/quickstart.md)
- [架构说明](architecture.md)
- [MCP 接入](mcp_integration.md)
- [示例代码](https://github.com/shibing624/dual-mem/tree/main/examples)

## 安装

```bash
pip install dual-mem
pip install dual-mem[all]   # + REST / CLI / MCP
```

详见 [Installation](getting-started/installation.md)。
