# 八层记忆模型（L0-L7）

dual-mem 采用与 hy_memory 一致的八层记忆分层（L0-L7）。本文件是该分层的**权威参考**；架构总览见 [`architecture.md`](./architecture.md)。

## 分层总览

```
事实/情境主线 (NORMAL_LAYERS, episodic)
  L1 RAW  →  L2 FACT  →  L3 SUMMARY  →  L5 KNOWLEDGE
画像/意图主线 (PROFILE / PROACTIVE, profile)
  L0 BASIC  →  L4 IDENTITY  →  L6 SCHEMA  →  L7 INTENTION
```

存储分界（概念对标 hy_memory 的 VDB/Graph 分界）：

- L0-L4：事实/画像主线。
- L5-L7：高层知识/图主线（Graph 层）。
- dual-mem 当前统一落库于 Chroma（VDB）+ SQLite，尚未拆分独立 Graph 存储；L6/L7 以普通节点形式存在。

## 逐层定义

### L0_BASIC_INFO — 基础信息层

结构化基础画像（姓名/年龄/所在地…）。

- **Category**：`profile`
- **产生方**：System1 `BasicProfile`（`dual_mem/agent/basic_profile.py`）
- **读路径**：归入 `PROFILE_LAYERS`，参与配额选中（identity 40% / schema 40% / 其余 20%）。

### L1_RAW — 原始对话层

写入即落的原始文本，Append-Only；写后转 `SHADOW`。

- **Category**：`raw`
- **产生方**：写入即落（`dual_mem/writer/memory_writer.py`）
- **读路径**：归入 `NORMAL_LAYERS`（legacy 路径召回；hybrid 路径不在 `VDB_RECALL_LAYERS`，仅 `NAVIGATIONAL` 意图走 `L1`）。

### L2_FACT — 原子事实层

从对话抽取的离散、版本化事实记录。

- **Category**：`fact`
- **产生方**：System1 Extractor（`dual_mem/agent/extractor.py`）→ fast-write 落 L2；Reconciler 可 `SUPERSEDE`。
- **读路径**：归入 `NORMAL_LAYERS` 与 `VDB_RECALL_LAYERS`，是语义/BM25 召回主力。

### L3_SUMMARY — 会话摘要层

长文本（≥500 字）压缩出的摘要。

- **Category**：`summary`
- **产生方**：System1 Summarizer（`dual_mem/agent/mem_agent.py`）
- **读路径**：归入 `NORMAL_LAYERS` 与 `VDB_RECALL_LAYERS`。

### L4_IDENTITY — 身份画像层

用户身份与长期偏好的核心画像。

- **Category**：`profile`
- **产生方**：System1 Extractor（identity 字段）→ fast-write 落 L4；Reconciler / System2 可整理。
- **读路径**：归入 `PROFILE_LAYERS`。

### L5_KNOWLEDGE — 知识图谱层（暂未实现）

实体/关系/主题类知识记忆（Graph 层）。

- **Category**：`knowledge`
- **产生方**：**无**。hy_memory 与 dual-mem 当前均未实现该层 producer；读路径 `NORMAL_LAYERS` 中保留此层，但不会有节点被创建。
- **状态**：占位层。后续若要落地，应在 Extractor 增加 `knowledge` 抽取字段，或在 System2 增加 `create_knowledge` 工具并纳入 `_S2_LAYERS`。

### L6_SCHEMA — 心智模型层

跨证据归纳出的抽象行为模式/叙事模板（Graph 层）。

- **Category**：`schema`
- **产生方**：System2 `System2Agent`（`dual_mem/system2/system2_agent.py`）+ `CrossDomainSweeper`（≥5 条基础 Schema 升维为核心 Schema）。
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
| dual | L0–L7（其中 L5 始终为空，无 producer） |
