# dual-mem Examples

可直接运行的端到端示例，使用**真实** OpenAI 兼容 API。

## 前置

把 API 配置写入 `~/.dual_mem/config.yaml`（参考仓库根 `config.example.yaml`），至少包含：

```yaml
llm_base_url: https://api5.xhub.chat/v1
llm_api_key: sk-...
llm_model: gpt-4o
embed_base_url: https://api5.xhub.chat/v1
embed_api_key: sk-...
embed_model: text-embedding-3-small
embed_dim: 1536
```

各 demo 通过 `mode=` 覆盖运行档位，存储目录用 `examples/.data/` 下的独立子目录（每次运行清空，可复现）。

## 运行

```bash
python examples/01_lite.py      # lite：纯向量召回，零 LLM
python examples/02_pro.py       # pro：抽取 + 演化链（Java→Python、上海→北京）
python examples/03_ultra.py     # ultra：digest 触发 System2，沉淀 L6 Schema / L7 Intention
python examples/04_rest_api.py  # REST：走真实 HTTP 契约
python examples/05_cli.py       # CLI：通过 dual-mem 命令行调用
```

> pro/ultra 会产生真实 LLM 调用（gpt-4o），有少量费用与数秒延迟。lite/REST/CLI 示例仅用 embedding。

## 并发写入（按需）

`add()` 是 async 的，多条**彼此完全独立**的记忆可以并发写入以缩短总耗时：

```python
import asyncio
results = await asyncio.gather(
    *(client.add(content=f, app_id=app, user_id=user) for f in facts)
)
```

注意取舍：

- 并发时多次 `add` 的 Reconciler 召回会重叠，**演化链/合并语义会变**——只适合互不取代、无需跨事实归纳的离散写入。
- ultra 的 System2 聚类依赖同主题离散事实各自落库；并发会把它们合并成一条，导致聚类样本不足、出不了 Schema。因此 `03_ultra.py` 刻意保持**串行**。
- 需要稳定演化链（如 `02_pro.py` 的 Java→Python、上海→北京）时，必须串行以保证顺序。
