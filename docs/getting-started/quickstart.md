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
| `mode` | 否 | `"system1"`（默认，写 L0–L4）或 `"dual"`（+ 显式 `digest()` 写 L6/L7）。可写在 YAML。 |
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
| `content="..."` | 单条陈述、日志、离线导入 | 整段由 Extractor 处理并决定是否沉淀。 |
| `messages=[{role, content}, ...]` | Agent 多轮对话（推荐） | L1_RAW 与 Extractor 都接收带角色标记的完整对话；宿主注入的 system 消息会先移除。 |

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
| `system1` / `dual` 短脚本 | 建议调用，释放存储句柄 |
| FastAPI / 长驻 Agent | lifespan shutdown 调一次；进程级单例一个 `MemoryClient` |

`aclose()` 不触发 digest；dual 应在业务确定的巩固点显式调用 `digest()`。

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
            f"extracted={write.extracted_count}  commit={write.commit_passed}"
        )
    finally:
        await memory.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

| 操作 | 建议 |
|---|---|
| `search` | 每轮用户输入前 |
| `add` | 会话结束一次（或每 N 轮 batch）；每次至少 1 次 Extract LLM，长文本可再触发摘要 |
| `WriteResult` | 读取 `commit_passed`；`False` 表示 Extractor 未提交 L0/L2+，`is_ephemeral=True` 表示纯临时内容 |

## dual 模式与 System2 触发

dual 写入只登记待处理 scope，不启动后台任务。由应用在会话结束、批量导入完成或其它明确边界调用 `await client.digest()`。

```python
from dual_mem import MemoryClient
from dual_mem.config import Settings

client = MemoryClient(settings=Settings(mode="dual", storage_dir="./.dual_mem_data"))
# ... await client.add(...) 若干次 ...
digest = await client.digest()
await client.aclose()
```

更多示例见 [examples/](https://github.com/shibing624/dual-mem/tree/main/examples)。
