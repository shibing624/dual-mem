# Changelog

## Unreleased

### 2026-10-18：单次写入、来源证据与有界资源

- **Jev 材料局部化**：每个候选只比较自身相似的当前 head/较早候选；无相关目标或 `NONE` 直接追加。默认 `reconcile_batch_size=1`。不再发送原文、representation/coverage 问题，也不把 revision、source/session 标识等控制面字段送进语义判断；正文、日期、类型、归属与历史值保留。CAS 与 provenance 仍在存储层。
- **真正移除传输重试**：Jev/embedding 不再循环或退避；LLM/embedding 的 OpenAI 客户端 `max_retries=0`；旧 extractor 的空结果重抽也删除。移除 `jev_retry_attempts`、`embed_retry_attempts`、`embed_retry_base_delay`、`extract_retry_on_failure`。旧 YAML 有这些键时应删除，不能用静默忽略掩盖配置错误。
- **召回统一尺度**：词法召回可以补充语义候选池外的精确匹配，补入候选保留真实 cosine；所有候选共同校准 BM25。profile 副路不再以原始 cosine 覆盖混合分，L3 的内部 RRF 分也不再直接与事实分比较。
- **来源证据（`raw_limit>0`）**：语义/词法召回结合命中记忆的 source links，L3 可沿全部引用证据找到原文。长来源返回有说话人标记、至多 3200 字符的相关片段，而不是头部截断；原始存储不改写。能补回抽取未保留的细节，但增加 QA 上下文长度，不能把它当作等 token 预算比较。
- **抽取与归属**：区分用户/任务记录、信息请求、未来计划和助手提供的具体信息；保留原因、条件与对象指代。profile 支持单属性 `attribute/value` 形式；助手陈述不进入用户 profile。`MemoryItem.owner` 与 formatter 的 `[user]` / `[assistant]` / `[derived]` 保留直接陈述和推导的区别。
- **版本感知 System2**：当前事实可携带真正旧版本的历史上下文，用于变化原因/时间线，不把旧值提升为当前，也不把旧 L3 推导再当用户证据。每条 draft 只校验引用材料；生成温度为 0；推导日期来自证据，ID 包含内容/类型/证据，避免返回旧内容时碰撞旧 ID。提交只对当前 head 做 CAS，旧版本仍保留。
- **可选 cross-encoder**：标准 `/v1/rerank` 接口，设置 `rerank_endpoint` 后对候选统一重排；默认关闭，失败直接抛出、不重试。重排时分数来自同一 cross-encoder，包括 raw 路，不再沿用其原始 0.5 缩放上限。
- **有界数据库资源**：新 HNSW 库使用 4 个线程；Kuzu 使用 4 个执行线程、128 MiB buffer pool、默认 64 GiB 最大映射（可通过构造参数增大）；关闭 VectorStore 同时关闭 Chroma 原生 client（上游引用计数保护同路径其它 client）。实测 384 核环境的多库进程可累积近 3 万线程；原 8 TiB/库地址预留也会让多库评估 mmap 失败。
- **条件性记忆**：S1 保留明确表达的体验评价、原因与情境；S2 推导有证据支持的驱动原则和适用边界，不把单一领域偏好直接推广到无证据的领域。这是统一生成/校验约束，不使用评测题目、选项或题型标签。实验中 LME 增益显著，但 PM 未达到预定 +10/+20pp，完整结果及泛化限制见 benchmark 变更日志。
- **验证**：445 passed / 1 skipped；有界线程设置下全套约 13 秒。Benchmark 结果与当前实验限制见既有 `exp/dual_mem_exp/benchmarks/CHAGNELOG.md`。

### 较早迭代记录（下列重试/复核/merge 行为已被上面的当前契约取代）

- **批内冲突一次性裁决，不再重裁**（2026-10-18）
  - 真实现象：LME 一个 session 的候选中，常有两条同时"取代"同一个当前 head（同一批抽取的两条派生事实、两个 session 汇报同一变更）。`_prepare_nodes` 会拒绝，于是整批重裁——而模型信息不变，重裁通常再撞一次，一次写入白付 2 份 JEV。
  - 现在 `_settle_target_claims` 在提交前一次扫过：同一 target 的多个取代按 `action` 置信度**留一条**，落选者原文写成独立记忆（内容不丢、旧 head 仍被取代）；指向"本批没产生记忆的 proposal"的悬空关系（RAW_ONLY/INVALID 导致的）直接剔除，剩余关系继续。
  - 一次内存扫描，**零额外模型调用**；实测 LME dual 34 次写入 0 次冲突报错（改前 13 小时内 33 次）。
- **删除 `ADD_MERGE` 动作（写路径不再做文本合并）**：合并要额外一次生成 + 一次无损校验，任何细节遗漏就整段失败（PM 实测约 5% 写入因此失败）。互补片段现在按 `ADD_SUPPLEMENT` 各自独立写入，主题/对象相同靠检索侧分组呈现。随之删除 `_generate_merges`、`collect_merged_timestamps`、`merged_timestamps` 字段与 `tests/test_reflector.py`。**回退**：如需恢复合并语义，从本 commit 前的 tag 取回 `_generate_merges` 与 `ADD_MERGE` 选项。
- **重写 System2（dual）：一批生成 → 一次校验 → 直接 append，不再走关系决策**（2026-10-18）
  - 根因：旧的 L3 派生记忆也要过一遍 System-1 的 `target` + `action` 决策，而 target 题要带**全部当前 head** 作为选项。LME 一个 scope 一百多个 head，每条派生记忆都重付一次；实测 LME dual 8.5 h 只跑完 2/36 题（system1 同期 8/36），其中 JEV 5.4k 次请求。
  - 现在：`System2Agent` 只做「按主题打包 facts → 一次 LLM 生成 → 一次 Jev 校验每条 draft → 直接作为独立 L3 节点 append」。派生记忆不取代 fact，选 target 是纯开销。重复的 draft 按 embedding 相似度（`s2_dedup_threshold`，默认 0.97）跳过。
  - 同一 scope 的批次按 topic 顺序串行；不同 scope 仍并发。
  - **实测**：4 条事实写入 + digest，system1 104 s / dual 115 s（旧的 dual 同场景要数分钟到数十分钟）。
- **删除全部"失败重试 / 降级保证"机制**。设计原则改为：宁可丢掉这一次推断，也不在 SDK 里堆重试，稳定性与延迟优先于覆盖率。
  - 删除 review 队列（`ReviewWorker`、`review_queue` 表、`pending_reviews()`、`needs_review` 字段与 `pending_review` 状态）。未定的关系（`NEED_CONTEXT`、多目标 `UNSURE`、低置信度）**当场**按独立 SUPPLEMENT 写入，之后不再回头复核。
  - 删除写侧降级：`degrade_to_supplement`、冲突计划的 `_degrade_conflicting`、锁内重裁。计划最多两轮（1 次规划 + 1 次带上轮拒绝原因的规划），仍失败就抛 `DecisionPlanError`。
  - 删除抽取重抽：`extract_coverage_retries` 去掉，抽取只跑一次；coverage 被拒仍提交候选，只打一条 warning。`REEXTRACT` 复核路径随之删除。
  - 删除 S2 失败表与二分：`s2_failures` 表、`record_s2_failure` / `quarantined_s2_facts` / `list_s2_failures`、`s2_max_attempts`。批次失败只记日志并丢弃；这些 fact 的 `s2_evidence_count` 仍为 0，下一次 `digest()` 自然会重新处理，不需要任何簿记。
  - 配置项移除：`reconcile_retry_attempts`、`extract_coverage_retries`、`review_max_attempts`、`s2_max_attempts`。**旧的 `~/.dual_mem/config.yaml` 含这些键会因 `extra="forbid"` 直接报错**，删除这四行即可（本机已改，备份为 `config.yaml.bak-1018`）。
  - 新增 `s2_dedup_threshold`（默认 0.97）。
  - 行为变化：`WriteResult.status` 不再有 `pending_review`；`MemoryDecision.review_reason`、`WriteResult.needs_review`、`MemoryItem.needs_review` 都已从模型与检索结果里删掉。REST schema 同步。
- **测试**：按新契约重写 `tests/test_system2_agent.py`（原 review/二分/隔离用例全部替换为"失败即丢弃"的断言）；`tests/test_review_regressions.py` 删除 review 队列相关 8 个用例。全套 435 passed / 1 skipped。
- **merge 生成漏条目时逐条降级**（2026-10-17 PM/LME 实验：7/237 次写入因此整段失败）：生成模型漏返回某个 `proposal_id`、返回空文本或只返回未知 id 时，只把这些 merge 写成原文 `SUPPLEMENT` 并标 `needs_review`（`merge generation omitted proposal`），其余 merge 照常做无损校验。原先抛 `merge generation lacks source coverage` / `returned empty content`，丢整段 source。未生成的 merge 不再花一次 JEV 校验。
- **同 scope 并发写入的 CAS 冲突在 SDK 内重试**：`reconcile_retry_attempts` 次都输给并发提交后，持 scope 锁再完整重裁最多 2 次（`_STALE_LOCKED_RETRIES`），仍失败才抛 `StaleDecisionError`（REST 409，语义不变）。LME 同一用户 2 路 session 并发灌入时，`targets changed concurrently` 原先约占写入的 2%。
- **测试**：`tests/conftest.py` 每个测试后释放 chromadb 进程级 `SharedSystemClient` 缓存。原先一次跑全套会在 ~48% 卡死在 chromadb Rust `Bindings(...)`（几百个未关闭的 PersistentClient 累积；单文件都能过）。现全套一次跑：458 passed / 1 skipped，约 53 s。
- **决策无法执行时降级，不丢整段 source**（2026-10-17，PM/LME benchmark 真实 Venus jev-1.13.0 发现）：
  - 冒烟中 3/8 个 PersonaMem chunk 写入整段失败（`multiple operations consume one target`、`Jev rejected lossy merge`），session 的全部记忆丢失。
  - 带反馈重裁 `reconcile_retry_attempts` 次后仍冲突：每个 target 保留**首个**决策，后续撞车的 proposal 原文写成独立 `SUPPLEMENT` 并标 `needs_review`（`_degrade_conflicting`），由 `digest()` 复核。
  - JEV 判 merge 有损（`INVALID`）：该 proposal 原文按 SUPPLEMENT 写入、旧 head 保持 current、关系入复核队列；原先抛 `DecisionContextError`。
  - MERGE 依赖的批内 proposal 最终没产生记忆：该 merge 降级为 SUPPLEMENT（原先 `DecisionPlanError`）。
  - 共用 `degrade_to_supplement()`。**效果**：冒烟 PM 题 ingest 失败 3/8 → 0/8。**回退**：无开关；冲突仍会先按反馈重裁。
- **超长 source 截断而不报错**：source（决策上下文，不是存储材料）超过 `jev_max_question_chars` 预算时保留首尾，coverage 题跳过（部分原文无法判覆盖）。原先抛 `source and proposals exceed the Jev context budget`，LME 长 session / PM 12-pair chunk 直接写入失败。proposals 本身超预算仍报错。
- **自部署 System One 兼容**：choice 答案缺 `confidence` 时取 `probabilities[choice]`（train-jev `trainjev.serve` 只返回 choice + probabilities，原先每次写入都校验失败）。
- ⚠️ 已知：train-jev jev-4b v0.2.0 不适合作写侧决策器——蒸馏语料（jev-distill-corpus-v3，655k 行）没有任何 memory-write choice 样本，8 条明确事实只落库 1 条（Venus jev-1.13.0 8/8）。实验用 Venus。
- ⚠️ Venus System One：单请求 JSON > ~64k 字符直接 422（`System One 请求校验失败`），单请求 150 题正常；限 30 QPM。`jev_max_request_chars` 默认 100k 会超，Venus 上需 ≤ 62k。
- **写入降级而不丢弃**：coverage 拒绝 / 空抽取被判漏抽时带反馈重抽（`extract_coverage_retries`，默认 1），用尽后仍提交候选并把 L0 来源排队复核；`NEED_CONTEXT`、多目标 `UNSURE`、低于 `reconcile_review_confidence`（默认 0 = 关）的关系先按独立 SUPPLEMENT 写入并标 `needs_review`。`digest()` 在 system1 / dual 都先跑复核，失败项 `review_max_attempts` 次后隔离。**效果**：benchmark 不再因单个 session 的决策失败丢整段记忆。**速度**：coverage 失败时多一次生成 + JEV。**回退**：`extract_coverage_retries=0`。
- **乱序灌入**：来源 `memory_at` 早于 target 时只能选 `ADD_HISTORICAL` 等不退役 head 的动作；旧报告作为非 latest 版本插到 head 后面。**效果**：并发 / 乱序 ingest 不会让旧 session 覆盖新状态。
- **JEV 预算**：target 选项改为短标签，完整节点只在 state 里出现一次（原先每个 head 出现两次，约 48 个 head 就超 60k 字符）。head 超过 `reconcile_candidate_limit` 或材料超预算时，按 embedding 相似度排序截断，只决定比较对象，不做写入门；取代原来的全量扫描 + 逐页 INCLUDE/EXCLUDE。**速度**：大 scope 每次 add 的 JEV 调用从 O(N/64) 降为常数；路由 embedding 写入时复用。
- `MemoryDecision.action` 原样返回 `RAW_ONLY` / `INVALID`，不再压成 `SKIP`；`WriteResult` 新增 `status`（committed / known / raw_only / pending_review）与 `needs_review`。
- 新增 `dual_mem.errors`，顶层导出 `DecisionContextError` / `DecisionPlanError` / `CoverageError` / `StaleDecisionError` / `IdempotencyConflictError`；REST 映射 409 / 422，`AddRequest` 支持 `request_id`，`AddResponse` 返回决策与状态。
- 抽取输出明确的 `valid_from` / `valid_until` / `temporal_scope`；`search(as_of=...)` 过滤过期的 TEMPORAL；命中附带一跳 `previous` 与仍有效的 `conflicts`（`include_relations`，默认开）；formatter 渲染有效期、上一值与冲突。
- `search(raw_limit=N)` 新增 L0 原文兜底路（`SearchMemories.raw`，分数 ≤0.5），默认关闭。REST / MCP 返回的 `memories` 多一个 `raw` 键。
- System2：分批前按 embedding 最近邻链排序让同主题证据同批；新增 `kind=preference`（隐式偏好推断，带证据）；被拒批次同次 digest 内二分隔离，单条事实失败 `s2_max_attempts` 次后隔离，scope 不再因坏批次反复失败。
- 幂等结果移到独立 `request_outcomes` 表，按 `request_outcome_ttl_seconds`（默认 7 天）在 digest 时清理。
- 修复（review）：同一批中两条 `HISTORICAL` 挂到同一个 current head 不再被当作「重复消耗目标」而让整次写入失败；退役目标仍只能被消耗一次，同一 head 不能在一批里既被退役又挂历史版本。
- 修复（review）：来源早于 target 时不再提供 `ADD_MERGE`（MERGE 会退役 head）；乱序动作集只剩 `HISTORICAL` / `CONFLICT` / `SKIP` / `INVALID` / `NEED_CONTEXT`。
- 修复（review）：复核意图在版本计划提交**之前**入队，计划 journal 同时记录 `needs_review`。进程死在提交与返回之间时，复核项仍在队列；队列行丢失时，同 `request_id` 重试从 journal 重放结果并补回复核项，不重新裁决。复核遇到尚未落库的临时版本会保留重试（写入可能仍在进行），对应写入以其它计划提交后视为无需复核。重新入队会清零 attempts。
- 修复（review）：复核重新抽取使用稳定的 `review:{id}` 作为 request_id，提交后、`resolve_review` 前崩溃时，下次 digest 从 journal 重放，不会因裁决不同撞上已存在的版本 id 而被隔离。
- 修复（review）：tool 消息（coding）路径的 `add(request_id=...)` 现在幂等：原文 id 与记忆 id 由 request_id 派生，结果与发布在同一 SQLite 事务写入 `coding_requests`，重复提交直接重放，复用 id 但输入不同抛 `IdempotencyConflictError`；中途崩溃留下的未发布预留行会在重试时清除。
- 修复（review）：coverage / REEXTRACT 复核与原写入的竞争。复核项记录发起它的 `request_key`；`digest()` 只在该请求已完成后才重新抽取，未完成（仍在提交或已失败等待重试）时保留重试。原写入提交后重新打开该项（attempts 清零），避免并发 digest 期间累积的失败把它推进隔离。同一 `request_id` 的重试若覆盖已通过，复核视为不需要。原文没有产生任何版本的写入（REEXTRACT、全部 RAW_ONLY/SKIP）也在 journal 记一条空计划，重试从 journal 重放，复核也能看到它已完成。
- `versions.db` 新增 `requests` 表（request_key → plan），`completed_request` 不再顺序扫描整个 journal；已有库在打开时从 plans 补齐。
- coding 路径的 `request_id` 冲突改抛 `IdempotencyConflictError`（REST 409，原先为 500）。
- coding 重放记录只写调用方显式传入 `request_id` 的请求（自动生成的 id 无法重放，写进去只会白占空间），并加 `ts` 列，由 `digest()` 按 `request_outcome_ttl_seconds` 一并清理。
- 默认 `storage_dir` 从 `./.dual_mem_data` 改为 `./.dual_mem_v4`。未显式指定目录时不再打开旧目录，旧数据不迁移。配置或参数仍指向旧目录时，打开照旧报 `unsupported memory schema` 且不改写。

## [0.1.3] — 2026-10-15

- Replace write-side semantic rules and similarity routing with batched JEV decisions.
- Keep four canonical layers and only system1 / dual modes; retain linked history.
- Remove legacy layers/statuses, deferred reconcile worker, nested config adapters and automatic migrations.
- Add per-attempt JEV latency and generation reasoning telemetry; hy3 smoke explicitly uses no_think.
- JEV target 选项从节点正文改为完整 `_context_node` JSON（layer、kind、revision、有效期、source_ids）。候选一多会先撞上 255 个选项或 `jev_max_question_chars`，抛 `DecisionContextError`，不会静默改成 SUPPLEMENT。action 说明把「另一次搬家」和「当前居住地变化」分开。
- 打开 `layer_schema_version != 4` 或非 `ACTIVE` 的目录直接报错且不改写。已有 `.dual_mem_data` 必须换新 `storage_dir`；旧 reconcile 队列不再排空。
- `llm_json_mode=true` 时，若 user 消息里没有单词 `json`，请求前会补上 `Reply with json.`。只写在 system 里不够：部分兼容网关只检查 input，会拒绝 `response_format=json_object`（System2 digest 原先只有 `Return {...}`）。
- JEV 关系裁决是 target-first，而且 action 集合双向闭合。有 target 只能选 `OVERRIDE` / `MERGE` / `NEGATE` / `CONFLICT` / `SKIP` / `DELETE` / `INVALID` / `NEED_CONTEXT`；无 target 只能选 `SUPPLEMENT` / `TEMPORAL` / `RAW_ONLY` / `INVALID` / `NEED_CONTEXT`。两段 instructions 分开，无 target 时不再提到 SKIP。`TEMPORAL` 和 `SUPPLEMENT` 不写 target，也不把 target 放进 CAS。动作和 target 不匹配抛 `DecisionContextError`，不重试；批内关系若指向最终 `RAW_ONLY` 的 earlier proposal，则作为依赖计划冲突重裁决。`DELETE` 只能指向已提交的 current head，不能归档尚未提交的 earlier proposal。`SKIP` 必须且只能有一个覆盖目标；多目标共同覆盖应生成可引用的 `MERGE`。主路径现在也严格校验注入 provider 的回答完整性、类型和选项。没有可选 target 时仍是一次 JEV；有可选 target 时多一次。coding 使用同一套两阶段，不再有 `:repair` 补问。
- `add()` 不再对长文本自动摘要。独立 `Summarizer` 已删除；`L3_DERIVED` 只由 dual 的 `digest()` 生成。
- 实验性 coding 写入与主路径使用同一套 JEV 动作和 target-first 约束。所有 target 关系都必须留在相同 workspace / branch / strict-session 边界，避免跨项目 `SKIP`、`CONFLICT` 或 `DELETE`；target choice 为 `NONE` / `MULTIPLE` 预留两个选项后，`CodingMemoryReconciler(candidate_limit=...)` 最大为 253。coding 的 `MULTIPLE` 表示需要逐 head 裁决，展开后必须唯一选中一个 head。自定义 `DecisionProvider` 只需实现协议声明的 `evaluate()`，请求大小限制由 provider 自己执行。`DELETE` 只归档当前 head，不另写一行。`SKIP` 必须指向当前 head。对外 `MemoryDecision` 带上真实 `targets`、`proposal_ids` 和 `update_type`。
- `list` / MCP `memory_list` / REST `GET /v1/memories/` / 集成工具增加 `include_history`（CLI 为 `--include-history`），默认仍只列当前版本。`update` 说明改为追加新版本，`delete` 说明改为归档。`digest` 只在 dual 生成 `L3_DERIVED`。来源标注从 `(来源: L1 …)` 改为 `(来源: L0 …)`。

本文件记录 **dual-mem SDK** 中会影响 **延迟、LLM 调用次数、召回/写入效果** 的变更。  
Benchmark 对比或线上行为异常时，优先查此表 + 对应 `Settings` / `config.default.yaml` 开关。

格式说明：

- **效果**：对准确率 / 召回 / 写入完整性的影响
- **速度**：对 ingest / search 耗时的影响
- **Trade-off**：刻意取舍
- **回退**：关闭或改回旧行为的配置

Benchmark 专用覆盖见 `exp/dual_mem_exp/benchmarks/backends/dual_mem.py`（与 SDK 默认值可能不同）。

---

## [JEV redesign · 2026-10-15 · branch]

- 四层 L0_RAW / L1_FACT / L2_PROFILE / L3_DERIVED；仅保留 system1、dual。
- JEV 批量 action/target/representation，已知与更新分开，六种 update_type。
  删除 novelty、weak similarity 路由、skip_llm、sync 与规则裁决分支。
- 所有 reconcile 追加并链接历史；update 返回新 ID，delete 归档，latest 默认读取。
- 默认并发 4，复用 HTTP；版本 CAS + 持久计划恢复 + 图投影待处理记录。
- System2 完整分页覆盖事实，processed evidence 与版本原子提交；来源更新使推导过期。
- 确定性生成 smoke 的真实 Venus JEV 最新四步裁决为 408–562 ms；真实 hy3 生成链
  四步写入为 12–28 秒，embedding 为确定性适配器。未做准确率或速度对比基准。
- JEV 的 target 选项携带完整节点 JSON，而不只是正文。材料过长或选项超过 255 时写入失败并保留 source，不降级成 SUPPLEMENT。
- 旧存储目录不能打开，也不会被迁移。换新的 `storage_dir`。
- 兼容性详见 docs/jev-memory-design.md。

## [Historical · pre-JEV notes]

> 下面是 JEV 重构前的未发版记录，不是当前 SDK 行为。其中的旧层名、`SHADOW`、Gate、`content_hash_dedup`、reconcile 队列和对应配置都已删除。

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
| **池内 BM25 置信度校准** | （内部） | 无额外存储查询 | BM25 分数同时考虑 query 覆盖率和候选池词项稀有度；精确稀有词仍可救回弱语义命中，但池内普遍词不再因 max-normalize 自动拿满分并穿过 `min_score` | 不引入 Trigger 模型或额外 LLM；部分词匹配的提升幅度低于旧行为 |
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

下面这段是 JEV 重构前的 benchmark 覆盖。这些键已经从 `Settings` 删除，不要抄进当前配置：

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

## 快速回退清单（当前 SDK）

| 现象 | 优先尝试 |
|------|----------|
| 打开存储报 `unsupported memory schema` | 换新 `storage_dir`。旧库不迁移、不改写，队列里的旧 reconcile 任务不会再跑 |
| `DecisionContextError`，或 JEV 报选项/字符超预算 | 把这次 source 拆成更小的写入。上限是 `jev_max_questions`、`jev_max_question_chars`，每个 target choice 最多 255 个选项 |
| 写入变慢 | 看 usage 回调里每次 JEV 的 `latency_ms`。默认并发 4 |
| 生成 JSON 解析失败又重试 | `extract_retry_on_failure: false` |
| vLLM 400 on json_mode | `llm_json_mode: false` |

---

## [0.1.2] — 2026-06

- 初始对外版本号；含 dual/system1 模式、hybrid 读路径、Gate LLM、System2 ReAct。
- 详见 git tag `v0.1.2` 与 `docs/architecture.md`。

## [0.1.1] — 2026-06

- 首个 dual-system memory SDK 发布（Gate + Extract + 快写 + 异步 reconcile）。

---

维护约定：任何 **默认配置变更**、**LLM / JEV 调用次数变更**、**截断或上下文预算变更**、**并发变更**，合并时写入文首 `Unreleased`。
