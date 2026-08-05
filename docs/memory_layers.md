# dual-mem 记忆层模型

dual-mem 实现 L0-L4、L6、L7 七个有明确 producer 和 consumer 的层。未实现的 L5 已删除，避免空枚举制造错误能力预期。

## 分层总览

```
事实/情境主线 (NORMAL_LAYERS, episodic)
  L1 RAW  →  L2 FACT  →  L3 SUMMARY
画像/意图主线 (PROFILE / PROACTIVE, profile)
  L0 BASIC  →  L4 IDENTITY  →  L6 SCHEMA  →  L7 INTENTION
```

存储分界（概念对标 hy_memory 的 VDB/Graph 分界）：

- L0-L4：写入 Chroma 向量库；SQLite 保存队列与审计信息。
- L6-L7：写入 Kuzu 图存储，并通过图向量检索召回。

## 逐层定义

### L0_BASIC_INFO — 基础信息层

结构化基础画像（姓名/年龄/所在地…）。

- **Category**：`profile`
- **产生方**：System1 `BasicProfile`（`dual_mem/agent/basic_profile.py`）
- **读路径**：作为 profile 结果参与 hybrid 召回。

### L1_RAW — 原始对话层

写入即落的原始文本，Append-Only；写后转 `SHADOW`。

- **Category**：`raw`
- **产生方**：写入即落（`dual_mem/writer/memory_writer.py`）
- **读路径**：raw 不进入默认语义召回池，避免与结构化记忆重复。

### L2_FACT — 原子事实层

从对话抽取的离散、版本化事实记录。

- **Category**：`fact`
- **产生方**：System1 Extractor（`dual_mem/agent/extractor.py`）→ fast-write 落 L2；Reconciler 可 `SUPERSEDE`。
- **读路径**：归入 `NORMAL_LAYERS` 与 `VDB_RECALL_LAYERS`，是语义/BM25 召回主力。

### L3_SUMMARY — 会话摘要层

超过配置阈值的长文本摘要；默认关闭，启用后默认阈值为 600 tokens。

- **Category**：`summary`
- **产生方**：System1 Summarizer（`dual_mem/agent/mem_agent.py`）
- **读路径**：归入 `NORMAL_LAYERS` 与 `VDB_RECALL_LAYERS`。

### L4_IDENTITY — 身份画像层

用户身份与长期偏好的核心画像。

- **Category**：`profile`
- **产生方**：System1 Extractor（identity 字段）→ fast-write 落 L4；Reconciler / System2 可整理。
- **读路径**：归入 normal，与 L2/L3 一起参与 hybrid 召回。

### L6_SCHEMA — 心智模型层

跨证据归纳出的抽象行为模式/叙事模板（Graph 层）。

- **Category**：`schema`
- **产生方**：System2 `System2Agent`（`dual_mem/system2/system2_agent.py`）。
- **读路径**：归入 `PROFILE_LAYERS`（hybrid 路径 `_PROFILE_LAYERS` 含 L6），图证据加成。

### L7_INTENTION — 前瞻意图层

用户未来待触发的具象意图（Graph 层）。

- **Category**：`intention`
- **产生方**：System1 intention 抽取（`dual_mem/agent/mem_agent.py`）+ System2 `s2_tools.create_intention`。
- **读路径**：归入 `PROACTIVE_LAYERS`，proactive 召回通道。

## 模式与写入层数

| 模式 | 写入层 |
| --- | --- |
| system1（默认） | L0–L4 |
| dual | L0–L4、L6–L7 |
