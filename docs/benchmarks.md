# 评测：dual-mem vs hy-memory vs mem0

一句话：**同一个模型、同一套题，只换记忆系统，dual-mem 在三个公开评测上都排第一，LoCoMo 上显著领先。**

| 记忆系统 | LoCoMo（308 题） | LongMemEval（24 题） | PersonaMem-v2（476 题） | 写入耗时（秒/段对话） |
|---|:---:|:---:|:---:|:---:|
| **dual-mem** | **74.68** | **91.67**（22/24） | **50.00** | **503** |
| hy-memory 2.0.4 | 66.56 | 70.83（17/24） | 43.49 | 2485 |
| mem0 | 65.58 | 83.33（20/24） | 47.90 | 808 |

写入耗时为 LoCoMo 每段长对话写入的中位数。

## 每个评测在测什么

| 评测 | 用户场景 | dual-mem 表现 |
|---|---|---|
| **LoCoMo** | 几百轮、跨月的长对话里，还记不记得对方说过什么、什么时候说的 | 74.7，领先 8–9 分；时间类问题 78 分，对手 61 / 22 |
| **LongMemEval** | 跨多次会话的信息抽取、更新和时间推理 | 24 题答对 22 题 |
| **PersonaMem-v2** | 根据长期积累的个人偏好做推荐和个性化回答 | 50.0，显著优于 hy-memory（43.5），高于 mem0（47.9） |

---

## 开发者可选阅读

<details>
<summary>评测协议、显著性、题型细分与复现方法</summary>

### 公平性约束

- 三个系统的抽取、作答和判分都用同一个 **Qwen3.8-27B**（FP8），embedding 都用 **bge-m3**。
- 题目集合在评测前冻结并记录 hash；每个系统从空库开始写入。
- 作答 prompt、检索条数（top-50）和判分 prompt 完全一致。
- dual-mem 用 SDK 默认配置，不针对评测调参。
- hy-memory 用 2.0.4 的 `ultra` 档（S1 + evolve）。
- mem0 用开源版 2.2.1。开源版不使用写入时间戳，评测里改为读取写入 metadata 中的会话时间来展示，和 mem0 官方评测的做法一致。

### LoCoMo

- 数据：`snap-research/locomo` 的 `locomo10.json`，共 10 段对话。
- 题集：按 Mem0 / EverMemOS 的公开口径去掉第 5 类对抗题，剩 1540 题；再按 Hy-Memory 论文的做法，每段对话固定抽 20%（种子 42），得到 **308 题**。
- 判分：与 EverMemOS 仓库中 LoCoMo judge 的 prompt 逐字相同，以 `CORRECT` / `WRONG` 判定。

| | 总分 | 对 dual 配对胜 / 负 | 符号检验 p | multi-hop | open-domain | single-hop | temporal |
|---|---:|---:|---:|---:|---:|---:|---:|
| dual-mem | 74.68 | — | — | 66 | 41 | 80 | 78 |
| hy-memory | 66.56 | 32 / 57 | 0.011 | 68 | 47 | 70 | 61 |
| mem0 | 65.58 | 36 / 64 | 0.007 | 80 | 47 | 79 | 22 |

- 写入耗时（每段对话中位数）：dual-mem 503 秒，hy-memory 2485 秒，mem0 808 秒。
- 已知短板：multi-hop 比 mem0 低 14 分，已列入 Roadmap。

### LongMemEval

- 题集：验证集固定 24 题（各题型均衡抽样）。
- 判分：统一的 LLM judge。
- 对 dual 的配对胜 / 负：hy-memory 0 / 5（p = 0.06），mem0 1 / 3（p = 0.63）。
- 样本较小：同一份存储重复读取的波动约 ±1 题，重新写入的波动约 ±3 题。

### PersonaMem-v2

- 题集：种子 42 的前 20 个 persona，共 476 道选择题。
- 对 dual 的配对胜 / 负：hy-memory 48 / 79（p = 0.007），mem0 57 / 67（p = 0.42，不显著）。
- 另有 20 个 persona（488 题）作为未参与调参的 holdout，2026-10-21 用当时的 SDK 测得：dual 52.66、mem0 49.80（未用会话时间修复，仅作参考）。
- 主表中 mem0 的 LongMemEval 和 PersonaMem-v2 结果是 2026-10-23 加上会话时间修复后重跑的。修复前分别为 75.00 和 50.00。

### 论文里的其他评测

Hy-Memory 论文还用了 Travel SPS、Shopping PS 和 CodeMemoryBench（CMB）。这三个评测的数据和评测代码目前都没有公开（论文、Hy-Memory 仓库和 HuggingFace 上都没有），暂时无法同场对比，公开后补上。

论文的绝对分数用的是别的写入模型和判分模型，不能和这里的数字直接比较，能比的是同场下的相对排名。

### 复现

评测工具在实验仓库 `dual_mem_exp/benchmarks`：

```bash
# LoCoMo：三个系统各跑一遍（20% 子样，种子 42）
python3 locomo/run.py --backend dual_mem  --output results/locomo/l20-dual
python3 locomo/run.py --backend hy_memory --output results/locomo/l20-hy
python3 locomo/run.py --backend mem0      --output results/locomo/l20-mem0

# 独立计分：只读结果文件，生成 scoreboard.json 和 EFFECT.md
python3 scripts/rsi_eval.py score
```

计分脚本不导入任何记忆系统，也不调用模型；缺答按错计。

</details>
