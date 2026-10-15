# 四层记忆模型

分层表达处理阶段；`kind` 表达语义，不再为每一种内容设置独立层。

| 层 | 内容 | producer | 读取 |
| --- | --- | --- | --- |
| `L0_RAW` | 原始输入证据 | writer，每次写入先保留 | `get` / `search_conversation` |
| `L1_FACT` | 显式事实、独立事件、明确意图 | 生成模型提供候选，JEV 裁决 | normal；`kind=intention` 可进入 proactive |
| `L2_PROFILE` | 状态、身份、画像维度 | 同一批 JEV 选择表示和关系 | profile；关闭 profile 路时进入 normal |
| `L3_DERIVED` | 跨证据模式、推导、摘要 | dual 的显式 `digest()` | profile / proactive；带实际 evidence IDs |

`system1` 写 L0/L1/L2；`dual` 共享完全相同的写入步骤，在 digest 时生成 L3。
显式意图放 L1；推导意图放 L3。`kind` 可取 fact、profile、intention、schema、summary。

版本由 `supersedes` / `superseded_by` 链接，退休表示保留原文、向量和时间，
`is_latest=false`。NEGATE 写入否定版本；TEMPORAL 保留基线并带有效期；CONFLICT
保留两侧并记录 `conflicts_with`。不存在按相似度把“更新”当作“已知”的写入门。

本次为破坏性变更：删除旧层名、旧状态与旧配置兼容，不自动迁移旧存储。
旧目录需离线导出后重新写入新目录。历史版本保持 ACTIVE，统一由 is_latest
及归档元数据区分当前可读性。Kuzu 是证据关系投影，VDB 决定当前可读性。
