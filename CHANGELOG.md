# Changelog

本文件记录 **dual-mem SDK** 中会影响 **延迟、LLM 调用次数、召回/写入效果** 的变更。  
Benchmark 对比或线上行为异常时，优先查此表 + 对应 `Settings` / `config.default.yaml` 开关。

格式说明：

- **效果**：对准确率 / 召回 / 写入完整性的影响
- **速度**：对 ingest / search 耗时的影响
- **Trade-off**：刻意取舍
- **回退**：关闭或改回旧行为的配置

Benchmark 专用覆盖见 `exp/dual_mem_exp/benchmarks/backends/dual_mem.py`（与 SDK 默认值可能不同）。

---

## [Unreleased]

> 当前 `__version__` 仍为 `0.1.2`；下列变更已在 `main`，尚未发版。

### 写入：Provider 先落 L1；读侧挂来源 id

- **`add_raw` / `distill`**：`add_raw` 只写 L1（embed，不 extract）。`distill` 对已有 L1 跑 extract，成功后 SHADOW 源 L1，不再写第二条 L1。`add()` 契约不变。
- **Hermes/OpenClaw `sync_turn`**：每轮先 `add_raw`；满 `write_turn_window`（默认 5）、闲置 `idle_timeout_sec`（默认 30s）、或 `on_session_end` / `shutdown` 再 `distill`。第 1 轮原话即可 `search_conversation`。
- **`MemoryItem.source_node_id`**：从 `custom` 带出。`format_memories` / `format_memories_for_prompt` 渲染 `(来源: L1 <8 位>)`。`get(L1 id)` 仍可读 SHADOW 原文。
- **效果**：宿主短会话不会等满 5 轮才有原文；模型能从 fact 跳回 L1。
- **回退**：`write_turn_window=1` 恢复每轮 extract；`idle_timeout_sec=0` 关闲置。

### 写入：注入隔离、SKIP 收回、原文检索、主题目录

- **回写剥离注入块**：`strip_memory_injection` 在 `client.add` 去掉 `<relevant-memories>` / `<user-profile>` / `<memory-tools-guide>` / `<topic-catalog>`。只剩注入块的 turn 不写 L1。避免 prefetch 被再抽成新记忆。
- **空 updates = SKIP**：reconcile LLM 明确输出 `{"updates": []}` 或 `[]` 时，整批新记忆视为无增量。dual 路径把对应 fast-write 标 `SHADOW`；解析失败仍 fail-open（空 ops，不收回）。`non_destructive` 不收回。也可显式 `op=SKIP`（`memory_id` / `new_index`）。
- **`search_conversation`**：只搜 L1_RAW（含提取后 SHADOW），不混进默认 QA `search`。REST `/v1/memories/search_conversation`、MCP/CLI/`conversation_search` 工具。与 `memory_search` 共享每轮 3 次上限。
- **稳定主题目录**：`load_profile_block` 在 L0 后追加 `<topic-catalog>`（ACTIVE L2/L4 tags + L6 标题，排序、与 query 无关）。
- **效果**：记忆不再自我增殖；重复 fast-write 可被收回；模型能核原话；system 尾部有可缓存的主题索引。
- **回退**：无开关。默认 `search` 仍不召回 L1。

### 写入：`update_type=MERGE` + 血缘 / 注入拆分

- **互补合并不再靠 prompt**：`fold_absorb_deletes` 把旧 Objective 2 的 `ADD + DELETE` 收成 `MERGE` + `supersedes` 链（旧节点 `SUPERSEDED`，不是 `SHADOW`）。`conservative` / `non_destructive` 不走这条改写。读侧 MERGE 链默认展开为 `(folded, still valid)`；OVERRIDE 仍隐藏旧值。
- **仅 MERGE** 写入 `custom.merged_timestamps`（新旧 `memory_at` 并集）。`format_memories` 在长度 > 1 时渲染 `(持续: YYYY-MM-DD → YYYY-MM-DD)`；OVERRIDE 替换链不画「持续」，避免把 Java→Python 说成从 Java 年代开始用 Python。
- **dual fast-write** 给 L2/L4 打 `custom.source_node_id`（指向源 L1）；System1 reconcile 路径原本已有。
- **integrations**：`load_profile_block` / `format_profile_block` 按 L0 list（与 query 无关）；`prefetch` 只注入 `memories.normal`。`system_prompt_block` 追加 Provider 内工具指南 + 缓存的 profile。`prefetch` 默认 3000ms 超时；`memory_search` 每轮最多 3 次，超限返回 `limit_reached`。
- **效果**：互补合并可沿演化链回溯；prefix cache 只在宿主把 `system_prompt_block` 放 system、`prefetch` 放 user 时生效。
- **回退**：prompt / `update_type` 无开关；`non_destructive` 仍剥离全部 `supersedes`（含 MERGE）。

### 新增：原生生态集成层（`dual_mem.integrations`）

- 新增 `dual_mem.integrations` 包，把 `MemoryClient` 适配到 5 个生态后端：`mcp`、`hermes`、`openclaw`、`agentica`、`claude_code`。
- 统一 `MemoryBackend`（MemoryClient 薄封装）+ `AsyncRunner`（同步 hook 驱动异步 client）+ `<relevant-memories>` 注入块格式化。
- 新增 `agentica` extra（`dual_mem.integrations.agentica`）与 `dual-mem-hook` 脚本（Claude Code `search` / `ingest` 子命令）。
- 纯新增对外接口，不改变现有 ingest / search 路径的 LLM 调用次数与延迟。

### 新增：coding 记忆子系统（`dual_mem.coding`，实验性）

- 针对含工具调用的工程对话，独立 extractor / judge / preproc / reconciler / writer / store 路径。
- 未接入 `dual_mem.__all__` 公共导出、无对应测试，标记实验性；稳定后再纳入 README 主推。

### Gate 子系统已整体移除（本批 · commit 484f221）

- `dual_mem/agent/gate.py` 与 `tests/test_gate.py` 删除；提交决策完全由 **Extractor 单次 LLM 输出**（`is_ephemeral` + 结构化记忆）驱动，不再有独立的 Gate LLM 调用。
- **LLM 调用次数**：每个 `add` 现在**固定 1 次 extract LLM**。旧 combined 模式下 PASS 也是 1 次；但旧 REJECT short-circuit 可 0 次，新模型 REJECT/ephemeral 仍付 1 次 extract。
- **已删除配置**：`combined_gate_extract`、`gate_heuristic_shortcircuit`、`gate_shortcircuit_novelty`、`gate_shortcircuit_relevance`、`gate_shortcircuit_reject_novelty`、`embed_merge_l1_gate`（含 `config.default.yaml` 对应段）。
- **公开 API 兼容**：`WriteResult.gate_passed` 字段保留；内部 `GateResult` 重命名为 `CommitResult`（`GateResult = CommitResult` 别名兼容）。
- **Reconsolidation**：不再做情绪/arousal 推断，仅刷新 `last_reactivated_at`（零 LLM）。
- **不可回退**：Gate 已删，无配置可切回旧行为；如需低价值内容跳过 extract LLM，需另加本地启发式预筛（TODO）。

### 写入路径（ingest）— LLM 次数与并发

| 变更 | 配置 | 速度 | 效果 | Trade-off / 回退 |
|------|------|------|------|------------------|
| **Gate 子系统整体移除** | 无（配置已删） | 每 `add` **固定 1 次 extract LLM**（无独立 gate LLM；旧 REJECT short-circuit 可 0 次，现 REJECT/ephemeral 仍付 1 次 extract） | 提交决策由 Extractor 输出（`is_ephemeral` + 结构化记忆）驱动 | 不可回退（Gate 已删）；低价值内容省 extract 需另加本地预筛（TODO） |
| **Extract ∥ Summary 投机并发** | `summarizer_enabled: true` 且 `len(content) ≥ summarizer_min_content_length` | PASS turn：`summary` 与 `extract` **并行**（≈ `max(T_extract, T_summary)`） | REJECT / ephemeral turn 可能 **白跑 1 次 summary LLM**（~5% REJECT × 长文本） | Summary 失败 **不阻断** L0/L2/L4（`try/except` 后 `summary=None`）。cancel 打 `summary cancelled after …` 日志。Benchmark 已 `summarizer_enabled: false`。回退：关 summary |
| **Summary 仅在 extract 产出持久化记忆后落库** | （逻辑，无单独开关） | 避免 ephemeral / 无结构化记忆 turn 写 L3 | — | 与上条配合；REJECT/ephemeral 时 `_cancel_summarize_task` |
| **Summarizer 阈值提高** | `summarizer_min_content_length: 1500`（原 500 → 800 → 1500） | 更少 summary LLM | 更短对话无 L3 | Benchmark：`summarizer_enabled: false` |
| **Extract 长输入：中间截断** | `extract_max_content_chars: 0`（SDK 默认不截断）；benchmark `10000` | 20k+ 字符 **避免 50–80s outlier** | **丢失中间段落信息**（head+tail 保留） | 旧行为：靠 `llm_context_window` **分块多次 LLM + merge**（见下「LLM 分块合并」）。截断 vs 分块：**截断 = 1 次 LLM 更快但丢中间；分块 = N 次 LLM 更慢但更全**。回退：`extract_max_content_chars: 0` 且依赖 chunk merge |
| **Extract JSON 失败 retry** | `extract_retry_on_failure: true` | 解析失败时 **+1 次 LLM**（~20s） | 降低「整 turn 22 条 fact 全丢」 | retry 用 `temperature=0.0` + **JSON-only 强化 prompt**（不强制 `json_object`，兼容 vLLM）。回退：`extract_retry_on_failure: false` |
| **多轮 messages 输入整形** | `extract_history_context_ratio: 0.7`，`extract_assistant_max_tokens: 200` | 仅当整段对话占用 > `0.7 × llm_context_window`（token 粒度）才截 assistant，长会话 prompt token 下降 | **不丢任何轮**（40 轮提交得 40 轮全在）；**user 全保留**；超阈值时 assistant（AI 的话）截到 200 token；短对话原样透传 | 仅作用于 `messages=`；`content=` 不受影响。回退：ratio 或 tokens 设 `0` |
| **配置单位统一为 token** | `summarizer_min_content_tokens: 600`、`extract_max_content_tokens: 0`、`extract_assistant_max_tokens: 200`、`extract_history_context_ratio: 0.7` | — | 全部阈值统一 token 粒度，内部 `× chars_per_token` 转字符比较。`extract_max_content_tokens`（原 `_chars`）= 最终输入硬上限，与历史整形**分层叠加不冲突** | 改名（删除旧 `_chars`/`_length`/`shape_threshold_ratio`，不兼容旧名）。benchmark：`extract_max_content_tokens: 4000` |
| **Content-hash 写路径去重** | `content_hash_dedup: true`，`content_hash_scope: session\|user` | 相同 content **跳过 extract/reconcile** | session scope：跨 session 不误命中；user scope：跨 session 命中率更高 | `session`（默认）按 `app/user/agent/session`；`user` 按 `app/user`。回退：`content_hash_dedup: false` |
| **L1 embedding 合并 gate 开关已移除** | 无（配置已删） | L1 节点 embedding 现由 `memory_writer` 统一 `embed_queued` | — | 旧 `embed_merge_l1_gate` 已移除，无此开关 |
| **Reconciler embed 批量化** | （内部） | reconcile 候选 embedding **batch + 并发 query** | — | — |
| **L0/L2/L4 + L3/L7 embed 合并** | （内部 `embed_batch`） | summary + intention 一次 embed | — | — |
| **演化链 supersede 原子化** | （内部 `VectorStore.mark_superseded`） | — | 锁内 `get→改 superseded_by/is_latest/status→update payload` 一次完成，**不重写 embedding**；并发 supersede 不丢更新 | 替换原 `get→改→upsert`（mem_agent / reconciler_worker / basic_profile 三处统一）。无配置开关 |

### 读取路径（search）— 延迟与排序

| 变更 | 配置 | 速度 | 效果 | Trade-off / 回退 |
|------|------|------|------|------------------|
| **Hybrid 五路 anchor `to_thread`** | `reader_mode: hybrid` | 阻塞 store IO 不卡 event loop | — | 回退：`reader_mode: legacy` |
| **Entity 路径：子串扫描 → BM25** | hybrid 内置 | 200 条候选池内 BM25，替代全表 substring | 关键词命中更准 | 仍限 `entity_pool_limit` 候选池大小 |
| **Entity BM25 分数映射** | （内部） | — | fusion 中 entity 不再 **全部 ~0.9** 压过 vector 0.4–0.7 | `score = 0.25 + 0.45 * bm25_norm`（`bm25_norm` 已 max-normalize）。旧：`min(0.9, 0.3+0.6*raw)` |
| **Fusion access 批量读** | （内部 `get_access_batch`） | 少 N 次 SQLite  round-trip | — | — |
| **Query understanding 只算一次** | （内部） | 避免 hybrid 路径重复 `understand()` | — | — |
| **Reconsolidation 限频** | `reconsolidation_min_interval_sec: 0`（SDK）；benchmark `60` | 同 user 搜索 **减少 hook 开销** | dual 模式 S2 reconsolidation 任务更稀疏 | 回退：`reconsolidation_enabled: false` 或 `min_interval_sec: 0` |
| **Reconsolidation 仅刷新时间戳（零 LLM）** | （内部 `_run_reconsolidation`） | 无额外 LLM | dual：召回后仅刷新 `last_reactivated_at`，**不再做情绪/arousal 推断、不再置 `custom.reactivation`**（Gate 移除后去此逻辑） | 非 ReAct，按设计轻量 |
| **Hybrid 按 `target_layers` 路由** | （内部，`reader_mode: hybrid`） | — | QU 建议层 ∪ 常驻 profile 层下传 anchor（旧文档误标"写死层列表"） | 回退：`reader_mode: legacy` |
| **Evolution chain 批量 `get_by_ids`** | （内部） | 链展开少 N 次 `vector.get` | — | — |

### Embed / LLM 基础设施（v0.1.2 后持续）

| 变更 | 配置 | 速度 | 效果 | Trade-off / 回退 |
|------|------|------|------|------------------|
| **LLM 超长 prompt：分块 + merge** | `llm_context_window`，`llm_completion_reserve`，`chars_per_token` | 超预算时 **N 次 LLM** 再 merge | 比硬截断 **更全** 但更慢 | Extract **额外** 可用 `extract_max_content_chars` **强制截断**（优先速度）。Gate / Reconcile 等仍走 chunk merge |
| **Embed 超长文本：分块 mean-pool** | `embed_max_tokens`，`chars_per_token` | 长文本 embed 正确性 | — | — |
| **Embed 内存 cache 扩大** | `embed_cache_size: 10000`（SDK）；benchmark `200000` | 重复文本 embed **命中缓存** | 内存占用 ↑ | 回退：减小 `embed_cache_size` |
| **Embed 写入侧攒批** | `embed_queue_batch_size: 32`，`embed_queue_window_ms: 200` | 并发 write 合并 embed 请求 | search 侧仍直接 `embed` | — |

### Benchmark backend 与 SDK 默认值差异

`exp/dual_mem_exp/benchmarks/backends/dual_mem.py` 相对 SDK 的典型覆盖（**刷榜时务必对齐**）：

```yaml
summarizer_enabled: false
extract_max_content_chars: 10000
content_hash_dedup: true          # session 级，PM/LME 命中率≈0
embed_cache_size: 200000
reconsolidation_min_interval_sec: 60
system2_trigger_mode: manual
persist_history: false
llm_json_mode: false              # vLLM Qwen 兼容
```

---

## 行为演进时间线（便于 git bisect）

| 时期 | 主题 | 代表 commit / 说明 |
|------|------|-------------------|
| v0.1.1 | Gate LLM 主路径、hybrid 读 | 基线：gate 与 extract **各 1 次 LLM** |
| fd63908 | 写侧 perf 第一批 | JSON mode、去 reconcile search-query、System2 多轮 |
| 07f6b4c | LLM/embed **分块 merge** | `chat_json_for_content` / `chat_text_for_content` chunk+merge |
| dcc8ae7 | P0/P1 ingest 优化包 | combined gate+extract、short-circuit、extract 截断/retry、summarizer 阈值、content_hash、读路径 BM25/to_thread |
| dc09d76 | P0 修复 | gate PASS 后再 summary（曾引入 extract∥summary 串行回归）、content_hash 加 session、BM25 entity 分数 |
| 47650c6 | 并发回收 | extract 前 **投机启动 summary**，REJECT/ephemeral **cancel** |
| 本批 | P0/P1/P2 收尾 + 历史整形 | messages 整形（assistant 截 500/最近 20 轮）、`content_hash_scope`、supersede 原子化、extract retry `temperature=0`+JSON-only、REJECT short-circuit、Kuzu `close()` 接入 `aclose()`；订正 target_layers / reconsolidation 过时文档 |
| 484f221 | Gate 子系统移除 | 删除 `gate.py`/`test_gate.py`；提交决策改由 Extractor 驱动；移除 `combined_gate_extract`/`gate_*`/`embed_merge_l1_gate` 配置；Reconsolidation 去情绪推断；`GateResult`→`CommitResult` |

---

## 快速回退清单（benchmark 异常时）

| 现象 | 优先尝试 |
|------|----------|
| ingest LLM 调用数异常 | 当前每 `add` 固定 1 次 extract LLM（Gate 已移除，无 gate 次数）；如需低价值内容跳过 extract 见本地预筛 TODO。其他查 `summarizer_enabled`、`extract_max_content_chars` |
| 长对话 fact 变少 / 丢中间信息 | `extract_max_content_chars: 0`（恢复 chunk merge）或增大截断上限 |
| REJECT/ephemeral 仍付 1 次 extract LLM | 设计如此（Gate 已删，无 short-circuit 可省）；如需省，加本地启发式预筛 |
| LME session 召回掉点 | 确认 `content_hash_dedup` 未跨 session 误命中；已含 `session_id` |
| search 排序异常 | `reader_mode: legacy`；或查 entity BM25 分数映射 |
| JSON 解析失败翻倍耗时 | `extract_retry_on_failure: false` |
| vLLM 400 on json_mode | `llm_json_mode: false`（benchmark 已设） |

---

## [0.1.2] — 2026-06

- 初始对外版本号；含 dual/system1 模式、hybrid 读路径、Gate LLM、System2 ReAct。
- 详见 git tag `v0.1.2` 与 `docs/architecture.md`。

## [0.1.1] — 2026-06

- 首个 dual-system memory SDK 发布（Gate + Extract + 快写 + 异步 reconcile）。

---

维护约定：任何 **默认配置变更**、**LLM 调用次数变更**、**截断/merge 策略变更**、**并发/scheduling 变更**，合并 PR 时请更新 `[Unreleased]` 一节。
