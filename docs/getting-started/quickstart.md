# 快速开始

## 同步（脚本 / REPL，推荐入门）

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
        print(m.content, m.evolution_chain)
```

`SyncMemoryClient` 在后台线程跑 event loop，**复用同一实例**时性能与 `MemoryClient` 一致；不要用 `asyncio.run()` 包每一次调用。

## 异步（FastAPI / asyncio Agent）

```python
import asyncio
from dual_mem import MemoryClient


async def main():
    client = MemoryClient(mode="system1", storage_dir="./.dual_mem_data")
    await client.add(content="...", user_id="alice")
    await client.aclose()


asyncio.run(main())
```

已在 async 事件循环内时，必须用 `MemoryClient` + `await`；`SyncMemoryClient` 会报错。

配置：`~/.dual_mem/config.yaml` 首次启动自动创建，填入 `llm_api_key` 与 `embed_api_key` 即可。

## 核心参数

### `MemoryClient(...)` 构造

| 参数 | 必填 | 含义 |
|---|---|---|
| `mode` | 否 | `"system1"`（默认，写 L0–L4）或 `"dual"`（+ 异步 System2 写 L6/L7）。可写在 YAML。 |
| `storage_dir` | 否 | 本地数据目录（Chroma + Kuzu + SQLite），默认 `./.dual_mem_data`。 |
| `settings` | 否 | 传入 `Settings` 对象，覆盖 YAML / 环境变量。 |
| `embed` / `llm` | 否 | 注入自定义客户端（测试 / 自建后端）；注入后跳过对应 `api_key` 校验。 |

### `add(...)` / `search(...)` 归属标识

| 参数 | 必填 | 含义 |
|---|---|---|
| `user_id` | **是** | 终端用户 ID，读写必须一致才能命中。 |
| `app_id` | 否 | 省略时用 `settings.default_app_id`（默认 `"default"`）。多产品共享同一服务时可显式传入。 |
| `app_ids` | search 否 | 省略时用 `[default_app_id]`。 |
| `agent_id` | 否 | 同一用户下细分 Agent 实例（多 Bot）。 |
| `session_id` | 否 | 会话 ID，可选更细隔离或审计。 |

配置项 `default_app_id`（YAML / `DUAL_MEM_DEFAULT_APP_ID`）可改默认命名空间。

### `content` vs `messages`

同一条 `add()` API，入参形式不同：

| 方式 | 适用 | 行为 |
|---|---|---|
| `content="..."` | 单条陈述、日志、离线导入 | 整段走 Gate → Extract。 |
| `messages=[{role, content}, ...]` | Agent 多轮对话（推荐） | L1_RAW 存完整对话；Extractor 看完整对话；Gate 向量新颖度只对 `role=="user"` 轮次计算（取 max）；LLM Gate 对整段打分；最后一条 assistant 作 Gate 上下文。 |

```python
await client.add(
    messages=[
        {"role": "user", "content": "我住在上海"},
        {"role": "assistant", "content": "好的，记住了"},
        {"role": "user", "content": "下个月要搬去北京工作"},
    ],
    user_id="alice",
)
```

## `aclose()` 要不要调？

| 场景 | 是否需要 |
|---|---|
| `system1` 短脚本 | 不必，几乎无操作 |
| `dual` + `per_write`（默认） | 进程退出前建议调；**不是**每次 add/search 后都调；**不会**等待已在跑的后台 digest |
| `dual` + `scheduled` | shutdown **必须**调，否则定时 loop 不会停 |
| FastAPI / 长驻 Agent | lifespan shutdown 调一次；进程级单例一个 `MemoryClient` |

不调的后果：`system1` 无影响；`scheduled` dual 可能停不干净；`per_write` 未完成的 System2 任务随进程退出丢弃（已 fast-write 的记忆不受影响）。

## Agent 集成（Agentica 多轮对话）

dual-mem 管**长期记忆**（跨会话），Agentica `WorkingMemory` 管**当前会话窗口**——两者配合，结构对齐 [Agentica `05_multi_turn.py`](https://github.com/shibing624/agentica/blob/main/examples/basic/05_multi_turn.py)：

- **每轮 `search`**：用用户本轮输入检索长期记忆，注入 `prompt_config.additional_context`
- **每轮 `agent.run`**：Agentica 自带 `add_history_to_context=True` 维护窗口
- **会话结束 `add(messages=...)`**：把 `working_memory.get_messages()` 批量写入 dual-mem

需已安装 Agentica（`pip install agentica`）并配置 LLM API key。

```python
import asyncio

from agentica import Agent, OpenAIChat
from dual_mem import MemoryClient
from dual_mem.retrieval.formatter import format_memories

APP_ID = "agentica"  # 可选；省略时用 default_app_id
USER_ID = "demo_user"


async def main() -> None:
    memory = MemoryClient(mode="system1")
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        add_history_to_context=True,
        instructions="你是一个友好的助手，请记住用户告诉你的信息。",
    )
    try:
        user_turns = [
            "我最喜欢的颜色是蓝色",
            "我喜欢吃川菜",
            "总结一下你了解到的关于我的信息",
        ]
        for user_msg in user_turns:
            ctx = await memory.search(
                query=user_msg,
                user_id=USER_ID,
            )
            block = format_memories(ctx.memories.to_dict())
            agent.prompt_config.additional_context = (
                f"以下是与用户相关的长期记忆，回答时可参考：\n{block}" if block else None
            )
            reply = await agent.run(user_msg)
            print(f"User: {user_msg}")
            print(f"Assistant: {reply.content}\n")

        session_messages = agent.working_memory.get_messages()
        write = await memory.add(
            messages=session_messages,
            user_id=USER_ID,
        )
        print(
            f"Session persisted: memory_id={write.memory_id[:8]}…  "
            f"extracted={write.extracted_count}  gate={write.gate_passed}"
        )
    finally:
        await memory.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

| 操作 | 建议 |
|---|---|
| `search` | 每轮用户输入前 |
| `add` | 会话结束一次（或每 N 轮 batch）；不要每轮都写（~2×LLM/次） |
| `WriteResult` | `gate_passed=False` / `is_ephemeral=True` 表示未沉淀 L2/L4 |

## dual 模式与 System2 触发

| `system2_trigger_mode` | 行为 | 示例 |
|---|---|---|
| `per_write`（默认） | 每次 write 后后台 fire-and-forget digest | `examples/02_dual.py` |
| `manual` | 仅入队，需 `await client.digest()` | `02_dual.py` 中也可设 `manual` |
| `scheduled` | 后台每 N 秒批量 drain；**shutdown 必须 `aclose()`** | `examples/05_scheduled_system2.py` |

```python
from dual_mem import MemoryClient
from dual_mem.config import Settings

client = MemoryClient(
    settings=Settings(
        mode="dual",
        storage_dir="./.dual_mem_data",
        system2_trigger_mode="scheduled",
        system2_schedule_interval_sec=300,  # 默认 300s；demo 可改短
    ),
)
# ... await client.add(...) 若干次，等待 interval 后 search ...
await client.aclose()  # scheduled 模式必须
```

更多示例见 [examples/](https://github.com/shibing624/dual-mem/tree/main/examples)。
