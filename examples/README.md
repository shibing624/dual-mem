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
