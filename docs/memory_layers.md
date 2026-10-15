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

旧层名作为读取兼容别名；枚举迭代只返回四层。加载 Chroma 时转换旧层元数据并
验证历史链：ACTIVE 的旧祖先也归一为非 latest，原文与向量不重写。旧 SHADOW/DELETED
改为 ACTIVE 的归档表示；不会恢复为当前事实。损坏或跨 scope 的链需要修复后再迁移。
旧图节点会复制到统一 VDB，Kuzu 保留为证据关系投影，VDB 决定当前可读性。
