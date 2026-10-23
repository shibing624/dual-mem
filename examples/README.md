# dual-mem Examples

可直接运行的端到端示例，使用真实 JEV、生成模型和 Embedding。建议按顺序看：
**system1 → dual → REST → CLI**。

## 前置

`MemoryClient` 需要 **JEV、LLM、Embedding 三套凭据**，缺 JEV key 会在构造时抛
`MissingCredentialsError`。首次构造会在 `~/.dual_mem/config.yaml` 写默认配置。

```yaml
jev_endpoint: https://api.typesafe.ai/v1/systemone
jev_api_key: ...
jev_model: jev-1.13.0

llm_base_url: https://api.openai.com/v1
llm_api_key: sk-...
llm_model: gpt-4o-mini

embed_base_url: https://api.openai.com/v1
embed_api_key: sk-...
embed_model: text-embedding-3-small
embed_dim: 1536
```

也可用环境变量（前缀 `DUAL_MEM_`）。yaml 里没有 `jev_api_key` 时，demo 会读
`VENUS_TOKEN`，并把 JEV endpoint 指到 `/v1/systemone`。
家目录配置里已经删除的键（例如 `system2_trigger_mode`）会被丢掉，不会让 `Settings` 报错。

```bash
export VENUS_TOKEN=...          # 没有 jev_api_key 时使用
export DUAL_MEM_LLM_API_KEY=sk-...
export DUAL_MEM_EMBED_API_KEY=sk-...
```

各 demo 用 `examples/.data/` 下的独立目录，每次运行清空。01 / 02 结束时
`await client.aclose()`。

## 运行

```bash
python examples/01_system1.py        # add(content/messages) + search + list
python examples/02_dual.py           # 同样的写入；digest() 只尝试生成 L3
python examples/03_rest_api.py       # REST。update 返回新 id，delete 归档当前版本
python examples/04_cli.py            # CLI 的 add / search / search-conversation / list
python examples/05_lifecycle.py      # add_raw、distill、原文检索、update/delete 演化链
python examples/06_coding_memory.py  # coding_enabled=True：工程排障 Tool 对话写入、方案演化与 search_coding
```

## 当前写入

- `add` / `search` 是 async。`WriteResult.decisions` 是这次 JEV 的 action 和 update_type。
  `memory_id` 是 L0_RAW。结构化版本在 `extra_node_ids`。
- `add(messages=[...])` 把整段对话交给生成模型和 JEV。`01_system1.py` 有例子。
- system1 和 dual 的 `add` 都在返回前完成版本写入。`digest()` 只属于 dual，
  用来写 L3_DERIVED，不会再裁决已经写下的 L1/L2。
- 默认 `list` / `search` 只看 `is_latest=true`。历史版本用 `list(include_history=True)`，
  或从检索结果的 `evolution_chain` 看。
- `search(intention_limit=0)` 时 proactive 为空。`include_derived=False` 关掉 L3 融合。
- `update` 追加一条 OVERRIDE，返回**新** id。`delete` 把当前版本标成非 latest，行还在。
  `05_lifecycle.py` 用这两步演示演化链，不依赖 JEV 把自然语言更新判成 OVERRIDE。
- `add_raw` 只写 L0。`distill` 对已有原文抽取，不再写第二条原文。
  默认 `search` 不返回 L0；`search_conversation` 只查 L0。
- `06_coding_memory.py` 演示 `coding_enabled=True` 下的工程记忆：`messages` 含 tool 调用时，
  JEV 判为工程场景后抽取 `task / solution / files`。三轮写入：两条独立经验（`ADD/SUPPLEMENT`），
  第三轮新方案替换第一轮（期望 `ADD/OVERRIDE`，JEV 也可能判 MERGE）。`search_coding` 只返回当前版本，
  `supersedes` 指向被替换的旧版本。消息使用标准 OpenAI `tool_calls` 格式，`tool_call_id` 与调用 id 一致。
  用法细节见 [快速开始 · Coding 记忆](../docs/getting-started/quickstart.md#coding)。
- MCP 工具直接对应上面这些 client 方法。
