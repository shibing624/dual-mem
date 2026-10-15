# JEV 记忆决策与迁移

可交互流程页：[JEV memory flow](design/jev-memory-flow.html)。实现只保留 system1 和 dual。

## 写入

1. 移除宿主注入块和 system 对话轮；保留 source 为 L0_RAW 原始证据。
2. 生成模型只抽取明确、原子化候选文本、日期及来源，不裁决是否已知或是否更新。
3. 读取同 scope 当前 latest 及真实 `history_versions`。材料较多时，用 JEV 分页判断
   INCLUDE/EXCLUDE；每条提议仍进入决策，没有向量相似度写入门，也没有弱相似度 bypass。
4. 同一 `state` 下批量提出来源 coverage 与每个候选的 action、representation、target 问题。问题彼此
   独立，不能假设目标问题知道动作问题的答案。协议组合必须通过执行约束。
5. ADD 联合 choice 是 `ADD_OVERRIDE` / `ADD_MERGE` / `ADD_SUPPLEMENT` /
   `ADD_TEMPORAL` / `ADD_NEGATE` / `ADD_CONFLICT`；另有 `SKIP`、`DELETE`、
   `RAW_ONLY`、`INVALID`、`NEED_CONTEXT`。后三项为控制结果，不产生有损写入。
6. 多目标才追加 INCLUDE 问题。MERGE 才用生成模型合成文本，并用 JEV 验证所有数字、
   独立事件与旧维度不丢失。正常写入为一次生成 + 一次 JEV；不为每项启动客户端。
7. 批量 embedding；检查 target revision、scope epoch、作用域、依赖顺序与重复消费；
   durable journal 后追加新节点、更新历史链接。临界区不包含正常模型 I/O；同 scope
   竞争失败时重决策，最后一次重试在 scope 锁内进行。不同 scope 并行。

SKIP 是“当前事实已覆盖”，RAW_ONLY 是“没有持久增量”。原文证据不是已存知识；
历史中出现过北京，不说明目前仍住北京。北京→上海→北京必须链接当前上海版本。
旧 novelty 门和 weak_candidate_score 是两个不同概念，但本设计两个都退出写入逻辑。

## dual / digest

`add()` 先完成同一 System1 决策和版本写入，再登记 cognition scope。
`digest()` 捕获队列 task IDs，同 scope 顺序处理，不同 scope 默认并行 4。
事实按照 `system2_max_prompt_facts` 和生成模型字符预算分批，孤立点也处理，
不使用 DBSCAN/cluster 去重作为证据覆盖。超预算材料保留 pending 并报错。

每批返回 `memories` 与 `processed_ids`；必须明确覆盖实际输入。JEV 校验推导与
覆盖声明，之后按同一版本协议写入 L3。只有实际成功处理的证据才标记完成。
来源更新后，引用该来源的旧推导变成非 latest，其余证据恢复 fresh；下次 digest 重建。
图投影待处理记录与版本同一提交；投影失败时，下次 digest 先恢复，已处理证据不会导致漏补图。
新到达的 task IDs 不会被旧水位确认。失败 scope 保留 pending，成功 scope 可独立确认。
旧 reconcile 队列经同一 JEV 协议迁移，不能直接清空。

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
  恢复未完成计划。非法向量在写日志前拒绝。跨机器共享目录/分布式事务不在此实现中。
- `persist_history` 控制附加审计库；版本内容和恢复日志始终保留。

## 配置与限制

JEV 可使用 `jev=` 注入 provider，或配置 `jev_api_key` / `jev_endpoint` / `jev_model`。
`Settings.from_dict({"jev": {...}})` 支持嵌套配置。Venus token 只在环境变量/注入中提供。
默认并发 4，HTTP 客户端复用，429/529/502/503/504 有限重试；按问题数量和字符预算拆包。
字符预算是保守配置，不是 tokenizer 实测 token 数。

材料上限默认 64、最多 220；每个 target choice 至多 255 个选项，提议特别多时应拆 source。
默认提议问题批量 12。材料/历史过长或动作与目标矛盾时显式报错并保留 source，
不会无依据地转成 SUPPLEMENT。连续 MERGE 传递全部祖先片段与来源，保持一批合并的一次生成/一次校验。JEV 的质量和完整真实生成链路速度仍须业务数据评测。

删除配置：reconcile_sync、reconcile_skip_llm、reconcile_weak_candidate_score、
reconcile_policy、reconcile_non_destructive、content_hash_dedup 以及 cluster 写入门控。
旧键由 Settings 的 unknown-key 兼容忽略，运行行为不会重新启用旧分支。

## 验证

```bash
python -m pytest tests/ -q
python ~/.agents/rules/check_ast.py .
python tmp/jev_smoke.py --offline
VENUS_TOKEN=... python tmp/jev_smoke.py --output /tmp/venus-results.json
```

Smoke 通过 MemoryClient 验证演化、已知判断、四个并行 scope 和显式 digest。
文本生成与 embedding 使用确定性适配器，JEV 是真实接口；这是接口/版本协议验证，
不等于业务 accuracy 的对比基准。

2026-10-15 最新实际结果：JEV 写入四步 561.5 / 408.3 / 556.5 / 500.8 ms；
四个独立 scope 并发写入 631.7 ms；5 个 scope digest 完成，全部 JEV HTTP 请求 18。
其中 digest 生成器输出带来源的摘要，并经真实 JEV 校验。另以真实 hy3 生成模型 + Venus
完成四步写入与 digest，写入约 12–28 秒；embedding 仍为确定性适配器。
尚无生产 embedding 联调、质量 A/B、稳定性/吞吐统计；单次演示不能证明效果或速度提升。

## Coding 工具来源

实验性 coding 路径同样先保留完整来源（包含 tool arguments/results）再调用语义服务。
正常请求为一次 scene JEV、一次候选生成、一次共享 state 的 coverage/action/target JEV；
服务字符预算和候选材料分页仍适用。UPDATE 对外映射为 ADD/OVERRIDE，追加新 ID。
SQLite 以 published/is_latest 决定可见性；所有向量写入确认与 scope CAS 成功后整批发布。
失败的候选保持不可见，旧 SQL head 保持可读；来源仍在 coding_sources/L0。
现阶段 coding 不支持相同 pending ID 自动回放，失败请求重新生成新 ID，旧 pending 可见性
始终关闭。SQLite 发布协议与主 SDK 的 versions.db 恢复协议不同；不宣称分布式事务。
直接 destructive update/delete 已拒绝，版本读取和完整 history_versions 均保留。

## 最终本地验证

2026-10-15：完整 pytest 为 378 passed / 1 skipped；28 条既有 FastAPI on_event
弃用警告。API AST 检查、compileall、diff whitespace、桌面/移动端流程页布局和交互均通过。
真实 Venus smoke 报告见 design/validation/venus-jev-smoke.json。
真实生成模型初次联调通过，但后续重复运行暴露空候选、coverage 拒绝和重复目标计划；
后两类非法提交均被阻止。重复目标允许有限次带约束反馈的 JEV 重决策；原文保留。
此联调记录需要与最终测试配置区分，不能将单次通过作为模型质量已验证的结论。

最终明确关系选项后，真实 hy3 + Venus 四步耗时 6.805 / 10.002 / 15.356 / 7.751 秒，
每步一次 JEV；状态回北京为 OVERRIDE、重复当前为 SKIP、只有一个 latest profile，
digest 也完成。核心状态断言通过，报告见 design/validation/venus-real-generation-final.json。
同次运行仍把两个独立搬家事件用 OVERRIDE 相连；原文历史保留，但事件关系质量尚未通过
评测。以上通过只指核心状态断言，不能推广为全部事实关系准确。保留负例报告以便后续评测。

复跑真实生成 smoke：配置 GENERATION_API_KEY / GENERATION_BASE_URL / GENERATION_MODEL
和 VENUS_TOKEN，再运行 `python tmp/jev_generation_smoke.py --output /tmp/generation.json`。
脚本以核心语义断言决定退出状态，不能用 HTTP 请求成功替代断言。
