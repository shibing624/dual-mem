# JEV 记忆决策与迁移

本地可交互流程页：`docs/design/jev-memory-flow.html`（Git 忽略，不随仓库分发）。实现只保留 system1 和 dual。

## 写入

1. 移除宿主注入块和 system 对话轮；保留 source 为 L0_RAW 原始证据。
2. 生成模型只抽取明确、原子化候选文本、日期及来源，不裁决是否已知或是否更新。
   原文写明有限有效期时，候选带 `valid_from` / `valid_until`（ISO 日期，按 UTC 整日展开）
   与 `temporal_scope`；格式不合法的边界被丢弃，记忆照常写入。
3. 读取同 scope 当前 latest 及真实 `history_versions`，作为比较材料。head 数超过
   `reconcile_candidate_limit`，或材料 JSON 超过 `jev_max_question_chars` 时，按与任一提议的
   最高 embedding 相似度排序并截断。**相似度只决定拿哪些旧版本来比较，不决定是否写入**：
   每条提议仍进入 JEV 决策，被截掉的 head 只是不参与这次比较。路由用的 embedding 在写入时复用。
   target 选项是短标签（`[layer/kind 日期] 正文前 240 字`），完整节点只在 `state.current` 出现一次。
4. 先问 coverage、representation 和 target。没有 current head、也没有更早的 proposal 时，
   action 跟这批一起问，仍然是一次 JEV。只要存在可选 target，action 放到第二阶段，
   并且只提供与 target 是否为空匹配的选项。
5. 有 target：`ADD_OVERRIDE` / `ADD_MERGE` / `ADD_NEGATE` / `ADD_CONFLICT` / `SKIP` /
   `DELETE` / `INVALID` / `NEED_CONTEXT`。来源 `memory_at` 早于所选 head 时（乱序灌入），选项换成
   `ADD_HISTORICAL` / `ADD_MERGE` / `ADD_CONFLICT` / `SKIP` / `INVALID` / `NEED_CONTEXT`：旧报告不能
   覆盖、否定或删除更新的 head。`HISTORICAL` 作为非 latest 版本插到 head 之后（新节点
   `superseded_by=[head]`，head 的 `supersedes` 追加它），head 保持 current。无 target：`ADD_SUPPLEMENT` / `ADD_TEMPORAL` /
   `RAW_ONLY` / `INVALID` / `NEED_CONTEXT`。`TEMPORAL` 不链接 baseline。`SUPPLEMENT`
   可以与其他 latest 共存，但不携带 target。两边的说明文字也分开，无 target 时不再提到 SKIP。
   `SKIP` 只能为一个 target 建立可引用别名；多目标共同覆盖时应选择 `MERGE`。
   `DELETE` 只能选择已提交的 current head；若批内关系选择的 earlier proposal 最终没有产生
   memory，整批按依赖计划冲突重裁决。
6. 多目标才追加 INCLUDE 问题。MERGE 才用生成模型合成文本，并用 JEV 验证所有数字、
   独立事件与旧维度不丢失。动作和 target 不匹配是 provider 契约错误，抛
   `DecisionContextError`，不把整批拿去重试。注入 provider 也必须精确回答全部问题，
   且 answer type 与 choice 必须属于对应问题。
7. 批量 embedding；检查 target revision、scope epoch、作用域、依赖顺序与重复消费；
   durable journal 后追加新节点、更新历史链接。临界区不包含正常模型 I/O；同 scope
   竞争失败时重决策，最后一次重试在 scope 锁内进行。不同 scope 并行。

SKIP 是“当前事实已覆盖”，RAW_ONLY 是“没有持久增量”。原文证据不是已存知识；
历史中出现过北京，不说明目前仍住北京。北京→上海→北京必须链接当前上海版本。
旧 novelty 门和 weak_candidate_score 是两个不同概念，但本设计两个都退出写入逻辑。
对外 `MemoryDecision.action` 原样区分 `ADD` / `SKIP` / `RAW_ONLY` / `DELETE` / `INVALID`。

## 单次写入，不复核、不重试

每次写入只做一次抽取、一次关系决策、一次提交；没有复核队列，也没有传输重试：

| 情况 | 处理 | `WriteResult.status` |
| --- | --- | --- |
| 抽取为空 | 不调用模型复查，来源保留为 RAW_ONLY | `raw_only` |
| 候选没有相似度 ≥ `reconcile_relation_threshold` 的 head/较早候选 | 直接 SUPPLEMENT（有效期时 TEMPORAL）追加，不调用 Jev | `committed` |
| `NEED_CONTEXT`、关系置信度 < `reconcile_review_confidence` | 当场按独立 `SUPPLEMENT` 写入 | `committed` |
| 同批多个候选取代同一 head / 指向未落库的较早候选 | 置信度最高者保留关系，其余独立写入；悬空关系剔除 | `committed` |
| `INVALID` 候选 | 不写入，决策保留在结果里 | 由其余决策决定 |

没有产生任何版本的写入（全部 RAW_ONLY/SKIP）也在 journal 记一条空计划，重试从 journal 重放。
`versions.db` 的 `requests` 表（request_key → plan）是对应索引，已有库打开时从 plans 补齐。

仍然抛错的只有：provider 契约错误（缺题、越界选项）、动作与 target 不匹配、来源本身超出 JEV
预算（`DecisionContextError`，REST 422）；外部写入改动了已链接的 head（`StaleDecisionError`，REST 409，
可原样重试）；`request_id` 复用不同输入（`IdempotencyConflictError`，REST 409）。异常从 `dual_mem` 顶层导出。

## dual / digest

`add()` 先完成同一 System1 决策和版本写入，再登记 cognition scope。
`digest()` 捕获队列 task IDs，同 scope 顺序处理，不同 scope 默认并行 4。

System2（v3，对齐 hy-memory 2.0.4）产出三类 L3：

- **schema**：当前 L1/L2 按 average-linkage cosine 聚类（`system2_cluster_distance`，超大簇按
  `system2_refine_distance` 再切，小于 `system2_min_cluster_size` 的簇不调用模型）。每簇一次生成，
  操作为 `create_schema` / `add_evidence`，证据须跨 ≥ `system2_min_sources` 个独立原文来源。
  DBSCAN 在单个用户的记忆上会链成一个簇，因此不用。
- **core schema**：基础 schema ≥ `system2_cross_domain_min_basics` 时，按去领域的行为抽象成组，
  每组至多一条跨领域 schema，证据继承成员。
- **persona catalog**：按 8 个领域从当前事实中选择原文（不改写），事实集合变化时才重建。

读取默认 `derived_context_limit=0`（schema 只写不读）、`catalog_context_enabled=True`。
生成失败只丢弃该产物，不重试；未提交的事实保持未处理。
来源更新后，引用该来源的旧推导变成非 latest，其余证据恢复 fresh；下次 digest 重建。
图投影待处理记录与版本同一提交；投影失败时，下次 digest 先恢复，已处理证据不会导致漏补图。
新到达的 task IDs 不会被旧水位确认。失败 scope 保留 pending，成功 scope 可独立确认。
旧 reconcile 队列和迁移 worker 已删除；新目录仅登记显式 digest 任务。

## 历史、恢复与公共 API

- `update(id, content)` 返回新 ID；旧 ID 仍能 `get` 原文，且 `is_latest=false`。
- `delete(id)` 归档当前表示，保留记录。`delete_bulk(confirm=True)` 为用户明确清除 scope，
  是独立的物理删除 API，不属于 reconcile。
- `list(..., include_history=True)` 查看历史；默认 list/search 只读取 latest。
- `WriteResult.decisions` 直接返回 action、update_type、targets、proposal_ids、source_ids、confidence。
- `MemoryItem` 暴露 layer、kind、revision、有效期、source_ids、supersedes、conflicts_with。
- `add(..., request_id=...)` 允许相同输入幂等重试；另一个事件必须使用另一 request_id。
  重试恢复 durable plan，重复文本不依赖 content hash 去重。
- `versions.db` 保存完整计划与审计事件；`versions.lock` 协调本机进程。进入存储读操作先
  先校验现存 schema 与全部待回放计划的 schema，再恢复未完成计划；旧格式拒绝且不改写。非法向量在写日志前拒绝。跨机器共享目录/分布式事务不在此实现中。
- `persist_history` 控制附加审计库；版本内容和恢复日志始终保留。
- 幂等结果存在 `cache.db` 的 `request_outcomes` 表（不再借用 profile 缓存），digest 时清理
  超过 `request_outcome_ttl_seconds` 的记录。

## 读取

- `search(as_of=...)`：显式有效期不包含该时刻的记忆被过滤（如已过期的 TEMPORAL）。
  benchmark 传问题时间；不传则不过滤。
- `search(include_relations=True)`（默认）：命中带一跳 `previous`（直接被取代的旧值，
  HISTORICAL 也会出现在这里）与仍然有效的 `conflicts`；formatter 渲染为 `(previously, 日期: …)` /
  `(conflicts, both unretracted: …)`，有效期渲染为 `(valid: 起 → 止)`。
- `search(raw_limit=N)`：L0 原文兜底路，语义 + BM25 做 RRF，分数压到 ≤0.5，结果在
  `SearchMemories.raw`，`flatten()` 会合并。默认 0 关闭。

## 配置与限制

JEV 可使用 `jev=` 注入 provider，或配置 `jev_api_key` / `jev_endpoint` / `jev_model`。
`Settings.from_dict({"jev_api_key": ..., "jev_endpoint": ...})` 只接受规范平铺配置。密钥放 `jev_api_key` 或 `DUAL_MEM_JEV_API_KEY`。
默认并发 4，HTTP 客户端复用，429/529/502/503/504 有限重试；按问题数量和字符预算拆包。
字符预算是保守配置，不是 tokenizer 实测 token 数。

材料上限默认 64、最多 220；每个 target choice 至多 255 个选项。选项文本是完整节点 JSON（layer、kind、revision、有效期、source_ids），不只是正文，所以更容易先撞上 `jev_max_question_chars`。提议特别多时应拆 source。
默认提议问题批量 12。材料/历史过长或动作与目标矛盾时显式报错并保留 source，
不会无依据地转成 SUPPLEMENT。连续 MERGE 传递全部祖先片段与来源，保持一批合并的一次生成/一次校验。JEV 的质量和完整真实生成链路速度仍须业务数据评测。

删除配置：reconcile_sync、reconcile_skip_llm、reconcile_weak_candidate_score、
reconcile_policy、reconcile_non_destructive、content_hash_dedup 以及 cluster 写入门控。
旧键及嵌套 mem0 配置不再兼容，Settings 校验直接报错。默认 `storage_dir` 是 `./.dual_mem_v4`，旧的 `.dual_mem_data` 不再使用。显式打开 `layer_schema_version` 不是 4、或状态不是 `ACTIVE` 的目录时抛 `ValueError` 并且不改写文件。

## 验证

```bash
python -m pytest tests/ -q
python ~/.agents/rules/check_ast.py .
python tmp/jev_smoke.py --offline
python tmp/benchmark_protocol_smoke.py
```

Smoke 通过 MemoryClient 验证演化、已知判断、四个并行 scope 和显式 digest。
文本生成与 embedding 使用确定性适配器，JEV 是真实接口；这是接口/版本协议验证，
不等于业务 accuracy 的对比基准。

真实验证使用确定性 embedding，生成与裁决分别计时。最新版数据见下方最终本地验证；
历史报告均保留在本地 docs/design/validation，不进入 Git。

## Coding 工具来源

实验性 coding 路径（`coding_enabled=True`）。`messages` 里出现 `role="tool"` 或
assistant `tool_calls` 才进入；其余写入走主路径。

1. **先留来源**：完整 messages（含 tool arguments / results，结构化 content 原样）写入
   `coding_sources` 与 L0_RAW，之后任何模型调用失败都不丢来源。
2. **scene**：一次 JEV choice 判 CODING / CHAT。CHAT 去掉 tool 消息后改走主路径。
3. **候选生成**：一次 LLM 调用，输出 `{"candidates": [...]}` 对象（JSON 模式只允许对象，
   不能是裸数组）。字段：`task`、`search_keys`、`solution`、`boundary_envs`（字符串）、
   `boundary_scope`、`confidence`；`files` 由 tool_calls 的路径参数结构化提取。
   只抽一次；不做 coverage 判定，与主路径一致。空候选直接 RAW_ONLY，不调用 JEV。
4. **target-first 裁决**：
   - 每个候选只看到**同一版本边界**内的 current head：相同 workspace / branch /
     boundary_scope，`strict` 还要求同一 session。其他边界的 head 不会作为选项出现。
   - 有可选 head 时先选 target（含 `NONE` / `MULTIPLE`；`MULTIPLE` 展开为逐个 INCLUDE，
     必须唯一确定一个 head），再按是否选中给出 action 选项；没有可选 head 时直接问 action。
   - head 超过 `candidate_limit`（默认 64，最大 253，给 `NONE` / `MULTIPLE` 留位）时先分页
     问 INCLUDE / EXCLUDE 缩小范围。
5. **执行**：`OVERRIDE` / `MERGE` / `NEGATE` 追加新 ID 并退役 target；`DELETE` 只归档；
   `SUPPLEMENT` / `TEMPORAL` / `CONFLICT` / `RAW_ONLY` 不退役任何 head。同一批多个候选可以
   替换同一个 head：head 退役一次，`superseded_by` 链接全部后继。同一 head 既被替换又被
   归档是矛盾裁决，在预留任何行之前报 `CodingDecisionError`。`INVALID` / `NEED_CONTEXT`
   同样报错，来源已保留。

**并发与发布**：snapshot → 裁决 → 发布按 `(app_id, user_id, agent_id)` 串行，同进程的并发
写会看到前一次的 head，不会撞 epoch；候选生成仍并行。SQLite 以 `published` / `is_latest`
决定可见性：候选行先以未发布状态预留，所有向量写入确认、scope epoch CAS 通过后整批发布。
另一进程共享同一数据库导致 epoch 变化时报 `CodingConcurrentWriteError`，重试即可。

**幂等**：传 `request_id` 时，raw id 与 memory id 由它派生，结果与发布同事务记录；重放
直接返回记录结果，输入不同报 `IdempotencyConflictError`。崩溃留下的未发布行在重试时清理。

**对外结果**：`WriteResult.memory_id` 是 raw id，`extra_node_ids` 是新版本 id；
`commit_passed` 与主路径同口径（追加了版本或归档了 head）；`decisions` 带 action、
update_type、targets、proposal_ids、source_ids。`search_coding` 只返回已发布的
`is_latest` 版本，历史保留在 SQLite（`supersedes` / `superseded_by`）。
直接 destructive update / delete 被拒绝。SQLite 发布协议与主 SDK 的 versions.db 恢复协议
不同；不宣称分布式事务。

## 最终本地验证

2026-10-15，main 合并后清理全部旧兼容路径。当时全量 pytest 为 390 passed、1 skipped。随后 coding 裁决、社区入口和文档又对齐了同一套 JEV 动作；以发布前重新跑的 `pytest tests/ -q` 为准。AST 检查通过。所有自动化测试 mock 模型与网络，真实接口验证仅由 tmp smoke 执行。

hy3 当前接口不能通过不传 reasoning_effort 关闭思考。实测只传
`extra_body={"thinking":{"type":"disabled"}}` 仍产生 314 reasoning tokens。
非思考验证显式使用：

```python
extra_body={"reasoning_effort": "no_think", "thinking": {"type": "disabled"}}
```

这里的 extra_body 是 OpenAI SDK 合并到请求顶层的扩展参数；全部五次 hy3 返回
reasoning_content 为空，reasoning token 计数未提供。不能把缺少计数当成服务器已报告 0。

下面这张表是 **target-first 之前** 的旧协议计时：当时每次写入只有一次 JEV。
现在只要存在 target 候选就会多一次 evaluation，调用数和延迟不能再拿来对比，需要重跑。

| 步骤 | hy3 API ms | JEV API ms | 端到端 ms | 调用数 hy3 / JEV |
| --- | ---: | ---: | ---: | --- |
| 我目前住在北京。 | 1234.2 | 494.1 | 1753 | 1 / 1 |
| 我从北京搬到上海了，现在住在上海。 | 1483.7 | 438.3 | 1953 | 1 / 1 |
| 我又搬回北京了，目前住在北京。 | 1355.7 | 465.0 | 1846 | 1 / 1 |
| 我目前住在北京。 | 1093.6 | 461.0 | 1568 | 1 / 1 |
| 显式 digest | 2819.3 | 1210.1 | 4139.0 | 1 / 3 |

旧协议下每次写入一次候选生成、一次批量 JEV 裁决。digest 那次含推导生成、来源校验及关系裁决，
还触发多目标选择，共三次 JEV。API ms 为各实际调用的延迟之和；并行时不等于墙钟
耗时。JEV 计时覆盖每次 HTTP 尝试，不包含排队和重试等待；端到端时间包含这些开销。

独立 JEV 真实接口 smoke 也是旧协议：四个 scope 并发写入 553.0 ms，5 个 scope digest
完成，合计 18 次 JEV 请求。生成与向量使用确定性适配器，不能与真实生成时延混算。
这组次数在 target-first 之后失效。

真实 smoke 六项断言全部通过：无思考内容、返回旧值为 OVERRIDE、重复为 SKIP、
两个独立搬家事件保持 current、三代画像保留、只有一个 current 画像。
报告留在本地 design/validation，不进入 Git。早期负例也保留；本次通过
不证明业务准确率或速度普遍提升。尚无生产 embedding 联调或业务质量 A/B。

```bash
python tmp/jev_smoke.py --offline
# 配置 GENERATION_API_KEY / GENERATION_BASE_URL / GENERATION_MODEL=hy3 / DUAL_MEM_JEV_API_KEY
python tmp/jev_generation_smoke.py --output /tmp/generation.json
```

脚本同时校验模型行为与版本语义，断言失败时退出非零；接口成功不等于 smoke 通过。
