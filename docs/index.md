# dual-mem

**All your agents. One Memory.** 让 AI 真正记住你：偏好变了它知道，换个工具它也知道。

- 🧠 **记得准**：像人一样整理记忆。事实、画像、习惯分开存；你改了主意，它会更新，不会混淆。
- 🔗 **记得通**：[MemHub](memhub.md) 让 Claude Code、Codex、Cursor、Hermes、Agentica 共享同一份个人记忆。
- 🏆 **同场领先**：同模型、同题集，LoCoMo 74.7，hy-memory 66.6，mem0 65.6，详见[评测](benchmarks.md)。

## 从哪里开始

| 我想… | 看这里 |
|---|---|
| 让我常用的 AI 工具共享记忆 | [MemHub](memhub.md) |
| 给自己的 Agent / 应用加长期记忆 | [安装](getting-started/installation.md) → [快速开始](getting-started/quickstart.md) |
| 接入 Agentica、Claude Code、Hermes 等框架 | [生态集成](integrations.md) |
| 通过 MCP 或 REST 接入 | [MCP / REST](mcp_integration.md) |
| 看效果对比 | [评测](benchmarks.md) |

## 三行代码

```bash
pip install dual-mem
```

```python
from dual_mem import SyncMemoryClient

with SyncMemoryClient() as memory:
    memory.add(content="我住在上海，最近开始不吃辣了。", user_id="alice")
    res = memory.search(query="给 alice 推荐一家餐厅", user_id="alice")
```

---

## 开发者可选阅读

想了解原理和实现细节，可以按这个顺序读：

1. [架构说明](architecture.md)：SDK 与 CLI、REST、MCP、Skill 的关系，以及存储层
2. [四层记忆](memory_layers.md)：L0 原文、L1 事实、L2 画像、L3 推导
3. [写入决策设计](jev-memory-design.md)：决策模型的动作与关系、版本链、并发与恢复
