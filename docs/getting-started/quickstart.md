# 快速开始

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
        print(m.content, m.evolution_chain)

    await client.aclose()


asyncio.run(main())
```

## 多轮对话

```python
await client.add(
    messages=[
        {"role": "user", "content": "我住在上海"},
        {"role": "assistant", "content": "好的，记住了"},
        {"role": "user", "content": "下个月要搬去北京工作"},
    ],
    app_id="my_app",
    user_id="alice",
)
```

Gate 对各轮 **user** 文本分别算向量新颖度（取 max）；LLM 对整段对话打分；Extractor 仍看到完整对话。

## dual 模式触发 System2

```python
client = MemoryClient(mode="dual", storage_dir="./.dual_mem_data")
# ... 多次 add ...
await client.digest()   # 或 system2_trigger_mode=per_write 自动触发
await client.aclose()
```

更多示例见 [examples/](https://github.com/shibing624/dual-mem/tree/main/examples)。
