# dual-mem Examples

可直接运行的端到端示例，使用**真实** OpenAI 兼容 API。建议按顺序通读：
**system1 → dual → REST → CLI**。

## 前置

dual-mem 需要 **同时配置 LLM 与 Embedding 两套 API key**（缺一者 `MemoryClient(...)` 即报错），
首次 `MemoryClient(...)` 会在 **`~/.dual_mem/config.yaml`** 自动创建默认配置（与仓库根 `config.example.yaml` 一致），填入 API key 即可：

```yaml
llm_base_url: https://api5.xhub.chat/v1
llm_api_key: sk-...
llm_model: gpt-4o

embed_base_url: https://api5.xhub.chat/v1
embed_api_key: sk-...
embed_model: text-embedding-3-small
embed_dim: 1536
```

也可用环境变量覆盖（前缀 `DUAL_MEM_`）：

```bash
export DUAL_MEM_LLM_API_KEY=sk-...
export DUAL_MEM_EMBED_API_KEY=sk-...
```

各 demo 通过 `mode=` 覆盖运行档位，存储目录用 `examples/.data/` 下的独立子目录
（每次运行清空，可复现）。async client demo（01 / 02）在结束时调用
`await client.aclose()`，让 dual 的后台 System2 loop 干净退出。

## 运行

```bash
python examples/01_system1.py    # system1：System1 抽取 + 演化链 + 多轮 messages 写入
python examples/02_dual.py       # dual：digest 触发 System2 ReAct + cross-domain
python examples/03_rest_api.py   # REST：`/v1/memories/...` 契约（system1 档）
python examples/04_cli.py        # CLI：dual-mem 子命令调用（system1 档）
```

> 所有 demo 都会产生真实 LLM + Embedding 调用（gpt-4o 等），有少量费用与数秒延迟。

## 当前 SDK

- `MemoryClient.add` / `search` 全部 **async**，返回 `WriteResult` / `SearchResult`
  等 dataclass，访问字段用 `res.memory_id`、`out.memories.normal[i].content`。
- `add(messages=[{"role": "user", "content": "..."}, ...])` 直接收多轮对话；Gate 取
  各轮 user 文本 novelty 的 **max**（`01_system1.py` 演示）。
- 没有"无 LLM"模式：缺失 `llm_api_key` / `embed_api_key` 时 `MemoryClient(...)` 直接抛
  `MissingCredentialsError`（fail-fast，比静默降级更直白）。
- `dual.digest()` 同时驱动 System2 ReAct 蒸馏和 Cross-domain Sweeper，可用
  `Settings(cross_domain_enable=True)` 显式开启（`02_dual.py` 演示）。
- 默认 reader 是 `hybrid`（QueryUnderstanding → 5 路 Anchor → GraphExpander →
  FusionScorer），`Settings(reader_mode="legacy")` 切回旧三路用于对比。

## 并发写入（按需）

`add()` 是 async 的，多条**彼此完全独立**的记忆可以并发写入以缩短总耗时：

```python
import asyncio
results = await asyncio.gather(
    *(client.add(content=f, app_id=app, user_id=user) for f in facts)
)
```

注意：

- 同一 `(app_id, user_id)` 的并发 `add` 已被客户端内部 lock **串行化**，不会撕裂演化链；
  但 dual 的 System2 聚类仍依赖同主题事实各自落库，并发写会让 reconcile 把它们
  合并成一条，**导致聚类样本不足、出不了 Schema**。
- 因此 `02_dual.py` 刻意保持串行；`01_system1.py` 也串行，以保证 Java→Python、
  上海→北京 的演化链顺序稳定。
- 并发只适合互不取代、无需跨事实归纳的离散写入（比如把不同用户的批量数据灌进库）。
