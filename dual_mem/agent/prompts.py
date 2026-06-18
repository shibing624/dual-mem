"""分层记忆 Agent 的 prompt 集合（中英双版，逐字来自 Hy-Memory 源码）。

每组提供中英两版，运行时按输入语言用 `pick` 选择，使 LLM 输出语言与输入一致。
占位符：
- EXTRACT/SUMMARY: {content}, {current_time}
- SEARCH_QUERY:    {new_memories}
- RECONCILE:       {current_time}, {existing_memories}, {new_memories}, {existing_tags}
"""

from dual_mem.providers.llm import is_chinese

EXTRACT_ZH = """你是一位专业的记忆分析专家。请从以下对话中提取关于用户的结构化信息。

对话内容:
---
{content}
---

当前时间: {current_time}

## 核心原则

1. **以用户为主体**: 所有记忆都以"用户..."开头，用第三人称描述。每条记忆必须是完整、自包含的语句——不要省略能赋予记忆意义的背景、条件或原因。
2. **保留所有细节**: 保留所有具体信息（地名、人名、品牌、数字等）。不要泛化或简化。
3. **语言一致性**: 输出语言必须与输入语言一致。中文输入→中文输出。
4. **时间处理**:
   - 如果对话中提到了时间（相对时间如"下周"、"昨天"，或绝对时间如"2025年3月"）：
     - 时间敏感内容（计划、具体过去事件、截止日期）：将时间自然嵌入到 `content` 中。如果提供了 `当前时间` 且引用的是相对时间，先将其转换为绝对时间（如"下周" + 当前时间 2026-04-24 → "2026年5月初"）再嵌入。
     - 非时间敏感内容（稳定偏好、人格特质）：保持 `content` 不包含时间。
   - 如果 `当前时间` 为空且内容包含相对时间表达：重写 `content` 使其不包含原始相对时间表达。
   - 如果没有任何时间表达：`content` 保持不变。

## 记忆层级

### identity (L4_IDENTITY)
用户的**个人属性、偏好、态度、观点、价值观和感受**——定义用户作为一个人的特质。

包括:
- 偏好和品味（喜欢、不喜欢、最爱）
- 态度、观点、价值观、信念
- 情感倾向和对事物的感受
- 人格特质和行为模式

每条 identity 记忆必须包含完整上下文。如果用户通过特定经历形成了某个观点，要包含该上下文。

**隐性偏好检测——读懂字里行间的含义**:
用户很少直接说"我喜欢X"或"我讨厌Y"。你必须留意态度暗示信号，并在发现时创建明确的 identity 记忆:
- 积极信号: 重新开始/恢复某活动、热情/兴奋的用词、主动投入、反复参与、热情洋溢的细节描述
- 消极信号: 负担/苦差事用词、失去兴趣、退缩、失望、回避
- 转变信号（关键）: "但现在..."、"最近..."、"以前...但是..."、"决定恢复/放弃"、脱离后重新参与

当出现这类信号时——即使是隐含的——你**必须**生成一条 identity 记忆，明确表述态度。例如:
- "我很快就失去了兴趣，感觉像是苦差事" → Identity: "用户对[活动]失去了兴趣，因为持续输出的压力让它变成了一种负担"
- "重新尝试这个爱好的兴奋感重新点燃了我的热情" → Identity: "用户重新找回了对[活动]的热情，觉得它又变得令人兴奋了"

**助手确认的洞察**: 如果助手对用户得出某个结论，而用户没有反对，则视为有效的隐含信息。

### facts (L2_FACT)
**客观事件、经历和事实发生的事情**——已经发生、正在发生或计划中的事。记录用户做了/将做什么，而非他们对此的感受。

包括:
- 过去的事件和经历
- 当前的活动和进行中的情况
- 未来的计划和安排
- 客观事实陈述（非观点）

### 基本结构化属性——由工具处理，不要作为 identity/facts 输出
稳定的结构化个人属性——**姓名、年龄、所在地、职业、雇主**——不作为 identity 或 fact 记忆输出。而是调用 `update_basic_user_profile` function-calling 工具，只传入对话中明确提到或更新的字段。未提及的字段不传。

- 不要生成仅复述这五个属性的 identity 记忆。
- 例外: 如果属性附带丰富的叙事上下文（如"用户2023年从工程师转为产品经理，因为想更贴近用户"），该叙事属于 `identity`；基本事实仍通过工具处理。

### 跨层重叠记忆
一段对话可能同时产出多条 facts + identity 记忆，内容可能有重叠——这是正常的。

### 合并——优先完整思想而非碎片
在每个层级（identity、facts）内，避免过度拆分。当对话中的多个句子描述同一件事的不同方面（同一个偏好、同一个事件、同一种态度、同一个计划）时，生成**一条**自包含的记忆来捕捉全貌，而不是多条碎片化的记忆。

只在各方面确实独立时（不同话题、不同维度）才拆分为不同记忆。当你犹豫两个要点是否应该合并时，倾向于合并——后续的 reconcile 可以拆分，但无法恢复因过度拆分而丢失的信息。

每条记忆应该有连贯的内在逻辑——一个完整的想法，而不是碎片。

## speculate 字段（适用于 identity 和 facts）

`speculate` 是一个可选的分析注释，用于模糊或间接信号的情况。当背后的动机、真实原因或隐含偏好在用户的话语中不明确时填写；否则为 null。

- **对于 identity**: 在态度信号模糊时解读隐含态度。
- **对于 facts**: 当真实原因不明确时解释事件为何发生，尤其是内在动机 vs 外在因素。
- 不要将 speculate 内容复制到 `content` 字段中——`content` 描述发生了什么/用户是什么样的；`speculate` 解释为什么。
- 对于清晰明确的陈述不要编造 speculate。

示例:
- "我最近开始做饭了"（identity）→ speculate: "用户可能因为生活方式变化或更多空闲时间而对烹饪产生了更多兴趣"
- "我辞职了"（fact）→ speculate: null（明确无歧义）
- "我取消了健身房会员"（fact）→ speculate: "用户可能因为时间安排而取消，而非对健身失去兴趣"

## Tags

每条 identity 和 fact 记忆必须包含 `tags` 字段——1到3个小写主题关键词（如 "音乐"、"旅行"、"工作"、"美食"、"健康"、"社交"、"技术"）。

## 输出格式

严格格式要求:
1. 输出必须是一个有效的 JSON 对象，包裹在以 ```json 开头、以 ``` 结尾的代码块中。
2. 代码块前后不要有任何文字。不要有代码块以外的 markdown。
3. 不要有解释、评论、道歉或思维链。
4. 字段类型必须严格匹配（字符串/数组/null）。未知或不适用的字段使用 `null`。
5. 如果 `identity` 或 `facts` 没有可提取的内容，使用空数组 `[]`——不要删除该键。

JSON 结构:

```json
{{
  "identity": [
    {{
      "content": "关于用户偏好/态度/观点的自包含记忆",
      "speculate": "对模糊信号的解读，如果明确则为 null",
      "tags": ["主题1", "主题2"]
    }}
  ],
  "facts": [
    {{
      "content": "关于事件/经历/计划的自包含记忆",
      "speculate": "如果原因不明确则解释为什么，否则为 null",
      "tags": ["主题1"]
    }}
  ]
}}
```

现在输出 JSON。"""

EXTRACT_EN = """You are an expert memory analyst. Extract structured user information from the conversation below.

Conversation content:
---
{content}
---

Current time: {current_time}

## Core Principles

1. **User as subject**: Every memory describes the user in third person. Use "The user ..." for English input, "用户..." for Chinese input. Each memory must be a complete, self-contained statement — never omit circumstances, conditions, or reasons that give the memory meaning.
2. **Preserve details**: Retain ALL specific information (place names, person names, brands, numbers, etc.). Do NOT generalize or simplify.
3. **Language consistency**: Output language MUST match the input language (English → English, Chinese → Chinese). Never translate content into a different language.
4. **Time handling**:
   - If the conversation references a time (relative like "next week", "昨天", or absolute like "March 2025"):
     - Time-sensitive content (scheduled plan, specific past event, deadline): embed the time into `content` naturally. If `current_time` is provided and the reference is relative, resolve it first (e.g. "next week" + current_time 2026-04-24 → "early May 2026") before embedding.
     - Not time-sensitive (stable preference, personality trait): keep `content` atemporal, do NOT include the time.
   - If `current_time` is empty AND content has a relative expression: rewrite `content` without the raw relative expression (describe atemporally).
   - If no time expression anywhere: `content` unchanged.

## Memory Layers

### identity (L4_IDENTITY)
User's **personal attributes, preferences, attitudes, opinions, values, and feelings** — what makes the user unique as a person.

Includes:
- Preferences and tastes (likes, dislikes, favorites)
- Attitudes, opinions, values, beliefs
- Emotional dispositions and feelings toward things
- Personality traits and behavioral patterns

Each identity memory must include full context. If the user formed an opinion through a specific experience, include that context.

**Implicit preference detection — read between the lines**:
Users rarely say "I like X" or "I hate Y" directly. Watch for attitude-laden cues and create explicit identity memories when you spot them:
- Positive: resuming/returning to an activity, passion/excitement words, voluntary effort, repeated engagement, enthusiastic detail
- Negative: burden/chore words, loss of interest, stepping back, disappointment, avoidance
- Shift (critical): "but now ...", "recently ...", "used to ... but ...", "decided to resume/quit", re-engagement after disengagement

When such a signal appears — even implicitly — you MUST emit an identity memory that explicitly states the attitude. Examples:
- "I lost interest quickly, it felt like a chore" → Identity: "The user lost interest in [activity] because the consistent-content pressure made it feel like a chore"
- "the excitement of revisiting this hobby has reignited my passion" → Identity: "The user has regained enthusiasm for [activity] and finds it exciting again"

**Assistant-confirmed insights**: If the assistant draws a conclusion about the user and the user does not object, treat it as valid implicit information. Keep the subject as "The user ..." even if the original conclusion was drawn by the assistant (e.g., "The user appears open to trying yoga after the assistant's suggestion").

### facts (L2_FACT)
**Objective events, experiences, and factual occurrences** — things that happened, are happening, or are planned. Captures *what* the user did/will do, NOT how they feel about it.

Includes:
- Past events and experiences
- Current activities and ongoing situations
- Future plans and scheduled events
- Objective factual statements (not opinions)

### Basic structured attributes — handled by tool, NOT by identity/facts
Stable structured personal attributes — **name, age, location, occupation, employer** — are NOT emitted as identity or fact memories. Instead, invoke the function-calling tool `update_basic_user_profile` with ONLY the fields that are clearly stated or updated in the conversation. Omit unmentioned fields.

- Do NOT emit an identity memory that merely restates one of these five attributes.
- Exception: if the attribute comes with rich narrative context (e.g., "The user became a product manager in 2023 after 5 years as an engineer because they wanted to work closer to users"), that narrative belongs in `identity`; the bare fact still goes via the tool.

### Overlapping memories across layers
A single conversation snippet may produce multiple facts + identity memories with overlapping content — this is expected and acceptable.

### Consolidation — prefer whole thoughts over fragments
Within each layer (identity, facts), avoid over-splitting. When several sentences from the conversation describe different aspects of the same underlying thing (same preference, same event, same attitude, same plan), emit ONE self-contained memory that captures the whole picture rather than several fragmentary ones.

Split into separate memories only when the aspects are genuinely independent (different topics, different dimensions of the user's life, different subjects). When in doubt about whether two points belong together, lean toward keeping them merged — downstream reconcile can split if truly needed, but cannot recover information dropped by over-aggressive splitting.

Each memory should have coherent internal logic — a complete thought, not a fragment.

## speculate field (applies to BOTH identity and facts)

`speculate` is an OPTIONAL analytical note for ambiguous or indirect signals. Fill it whenever the underlying motivation, real cause, or hidden preference is NOT explicit in the user's words; otherwise leave it null.

- **For identity**: interpret implicit attitude signals when ambiguous.
- **For facts**: explain *why* an event happened if the real cause is ambiguous, especially intrinsic vs extrinsic motivation.
- Do NOT duplicate speculate content into the main `content` field — `content` describes what happened / what the user is; `speculate` explains why.
- Do NOT fabricate speculate for clear, explicit statements.

Examples:
- "I've been cooking more lately" (identity) → speculate: "The user may have developed more interest in cooking, possibly due to lifestyle changes or more free time"
- "I quit my job" (fact) → speculate: null (explicit, no ambiguity)
- "I canceled the gym membership" (fact) → speculate: "The user may have canceled due to schedule constraints rather than loss of interest in fitness"
- Assistant suggests yoga, user doesn't object (identity) → speculate: "The user may be open to yoga but has not yet confirmed"

## Tags

Each identity and fact memory must include a `tags` field — 1 to 3 lowercase topic keywords (e.g., "music", "travel", "work", "food", "health", "social", "technology").

## Output contract

Strict formatting rules:
1. Output MUST be a valid JSON object wrapped in a fenced code block that starts with ```json and ends with ```.
2. NO prose before or after the fenced block. NO markdown other than the single fenced block.
3. NO explanations, comments, apologies, or chain-of-thought.
4. Field types MUST match exactly (strings / arrays / null). Unknown or not-applicable fields use `null` (never omit them, never use `"null"` string).
5. If there is nothing to extract for `identity` or `facts`, use an empty array `[]` for that field — do NOT drop the key.

JSON shape:

```json
{{
  "identity": [
    {{
      "content": "self-contained memory about user's preference / attitude / opinion",
      "speculate": "interpretation of ambiguous signals, or null if clear-cut",
      "tags": ["topic1", "topic2"]
    }}
  ],
  "facts": [
    {{
      "content": "self-contained memory about an event / experience / plan",
      "speculate": "why it happened if ambiguous, else null",
      "tags": ["topic1"]
    }}
  ]
}}
```

Output the JSON now."""

SEARCH_QUERY_ZH = """你是一个搜索查询生成器。给定一组新提取的记忆，生成一组简短的搜索查询，用于在向量数据库中找到相关的已有记忆。

目标是最大化召回率——即使措辞差异很大，也要找到与新记忆语义相关的已有记忆。

## 新记忆:
{new_memories}

## 指引

生成的搜索查询应覆盖:
- 记忆中提到的关键主题、实体和主题
- 核心概念的改写或抽象版本
- 用户记忆库中可能存在的相关概念

输出一个 JSON 字符串数组（5-15条查询，简短聚焦）:

["查询1", "查询2", "查询3", ...]

只输出 JSON 数组，不要有其他文字。"""

SEARCH_QUERY_EN = """You are a search query generator. Given a list of newly extracted memories, generate a set of short search queries that can be used to find related existing memories in a vector database.

The goal is to maximize recall — find existing memories that are semantically related to the new memories, even if the wording is very different.

## New memories:
{new_memories}

## Instructions

Generate search queries that cover:
- Key topics, entities, and themes mentioned in the memories
- Rephrased or abstracted versions of the core concepts
- Related concepts that might exist in the user's memory store

Output a JSON array of query strings (5-15 queries, short and focused):

["query1", "query2", "query3", ...]

Output JSON array only, no other text."""

RECONCILE_ZH = """你是一个记忆管理系统。你的任务是将一批新记忆整合到已有的记忆库中，同时保持记忆库的整洁、可检索和信息无损。

当前时间: {current_time}
（`当前时间` 仅供参考。不要仅因为某条已有记忆的 `memory_at` 时间较久远就认为它已过时——时间久远本身不是操作它的理由。）

## 已有记忆
{existing_memories}

每条已有记忆包含以下字段:
- `memory_id`: 稳定标识符（仅出现在链头节点上——这些是你可以在操作中引用的唯一 ID）。
- `content`: 存储的文本。
- `memory_at`: 记忆发生的时间戳（ISO 格式）。可能为 `null`（时间未知）。
- `layer`: `l2_fact` 或 `l4_identity`。
- `tags`: 主题关键词。

部分记忆可能包含 `history_versions` 字段——这是 supersedes 链（见下方解释）。

## 理解 supersedes 链

链代表**某个特定维度上的状态演进**——用户在某件事情上的立场随时间发生了变化。

给定链 A → B → C（A 是最新的链头，C 是最旧的）:
- 三个节点都在讨论**同一个维度**（如"首选饮品"），但各自陈述了不同的主张。
- A 是该维度上的**当前真相**。
- B 和 C 是**历史**——它们在该维度上的主张已经过时。
- 但是，B 和 C 可能包含**其他信息**（在不同维度上），这些信息仍然有效且有用。

示例:
- C: "用户住在北京，从事软件工程师工作。"
- B: "用户因新工作搬到了上海。"（在居住地维度上取代 C——不可能同时住在两个城市）
- A: "用户搬到了东京。"（在居住地维度上取代 B）

这里只有居住地维度在演进（一个人同时只能住在一个城市）。C 中"从事软件工程师工作"的事实仍然有效——它从未被否定。

当你看到已有记忆上的 `history_versions` 时，在做决策之前先阅读完整链以理解完整历史。

## 待整合的新记忆
{new_memories}

## 操作原语

你有两种操作原语:

### ADD — 创建一个新记忆节点
- `op`: `"ADD"`
- `content`: 记忆文本。第三人称（"用户..."）。自包含。
- `layer`: `"L2_FACT"` 或 `"L4_IDENTITY"`。
- `supersedes`: 此新节点在争议维度上取代的已有 `memory_id` 列表。默认 `[]`。被引用的节点作为历史版本保留在链上。
- `tags`: 1-3个小写主题关键词。**优先从下方已有标签列表中选取，仅在现有标签都不合适时才新增。**

### DELETE — 逻辑删除一条已有记忆
- `op`: `"DELETE"`
- `memory_id`: 要删除的已有节点。仅在有对应的 ADD 已完全吸收其内容时使用。

## 已有标签（优先复用，避免同义重复）
{existing_tags}

## 目标

### 目标 1 — 为状态变化建立矛盾链
当新记忆和已有记忆在**同一维度**上的声明**不能同时为当前真实状态**时，形成链: 生成一个 ADD，其 `supersedes` 指向已有的链头节点。

关键测试: "如果读者问'用户当前在维度 X 上是什么状态？'，旧节点会给出错误答案吗？" 如果是 → 取代它。

不要用 `supersedes` 来处理补充、细化或累积——那些归入目标 2。

### 目标 2 — 合并碎片化的记忆
当记忆描述同一主题但兼容（无矛盾）时，合并它们:
- 生成一个 ADD，包含合并后的内容（`supersedes: []`）。
- 对每个被吸收的已有节点生成 DELETE。
- 合并后的内容必须保留所有来源的每一条有意义的事实。

### 目标 3 — 粒度控制
每条记忆应代表一个连贯的想法，具有完整的内在逻辑。避免两个极端:
- 过于细粒度: 缺乏独立上下文的碎片化句子
- 过于粗粒度: 同一主题膨胀过大（例如超过约2000字符），在一个节点中塞入过多信息。此时应将过大的节点拆分为多条更细维度的节点——每条覆盖一个独立子方面——同时确保拆分过程中信息零丢失。

犹豫时，倾向于稍粗一些（将相关方面放在一起），而非过度拆分。

### 目标 4 — 不要动不需要改变的东西
你选择不操作的记忆保持不变。不要对它们生成操作。

## 绝对前提——信息无损保留

**任何来源——新的或已有的——的任何信息在你的操作执行后都不得丢失。** 每条有意义的事实都必须保持可检索: 作为链节点的内容、作为合并 ADD 的一部分、作为独立 ADD、或在原始节点中保持不变。

## 硬约束

- 每个 ADD 的 `content` 必须自包含且为第三人称。语言必须与输入记忆一致。
- `memory_at: null` 表示时间未知。绝不要编造时间线；绝不要以"年代久远"为由取代记忆。
- 优先同层操作（L6↔L6，L2↔L2）。跨层仅在语义意义确实转移时使用。
- 输出语言与输入记忆语言一致。

## 验收自检（在输出前心理运行）

1. **信息无损**: 每条新记忆和你正在取代或删除的每条已有节点中的每个事实，都在最终状态的某处存在。
2. **碎片已清理**: 没有两个非链节点在明显同一主题上作为独立碎片存在（合并会改善可检索性的情况）。
3. **没有无补偿的 DELETE**: 每个 DELETE 都有一个吸收其内容的对应 ADD。

## 输出格式

顶层是一个 JSON 分组数组。每组是一个逻辑更新:

```
{{"reason": "<简短描述>",
  "ops": [ <操作>, <操作>, ... ]}}
```

格式参考:

```json
[
  {{
    "reason": "<为什么需要这组操作>",
    "ops": [
      {{"op": "ADD", "content": "用户...", "layer": "L4_IDENTITY",
        "supersedes": [], "tags": ["..."]}},
      {{"op": "ADD", "content": "用户...", "layer": "L4_IDENTITY",
        "supersedes": ["<existing_id>"], "tags": ["..."]}},
      {{"op": "DELETE", "memory_id": "<existing_id>"}}
    ]
  }}
]
```

## 输出约定（严格）

1. 输出必须是一个单独的 JSON 数组，包裹在以 ```json 开头、以 ``` 结尾的代码块中。
2. 代码块外不要有任何文字、markdown、道歉或思维链。
3. 如果不需要任何变更，在代码块中输出 `[]`。
4. 字段类型必须严格匹配: `supersedes` 和 `tags` 是数组（为空时用 `[]`）。

现在输出 JSON 数组。"""

RECONCILE_EN = """You are a memory management system. Your task is to integrate a batch of new memories into the existing memory base while keeping it clean, retrievable, and losslessly informative.

Current time: {current_time}
(`current_time` is provided for reference only. Do NOT infer that an existing memory is outdated solely because its `memory_at` is far in the past — age alone is not a reason to touch it.)

## Existing memories
{existing_memories}

Each existing memory has these fields:
- `memory_id`: stable identifier (only present on head nodes — these are the only IDs you can reference in ops).
- `content`: the stored text.
- `memory_at`: timestamp when the memory occurred (ISO format). May be `null` (time unknown).
- `layer`: `l2_fact` or `l4_identity`.
- `tags`: topic keywords.

Some memories may include a `history_versions` field — this is the supersedes chain (explained below).

## Understanding supersedes chains

A chain represents **state evolution on one specific dimension** — the user's stance on something changed over time.

Given a chain A → B → C (where A is the latest head, C is the oldest):
- All three nodes talk about the **same dimension** (e.g., "preferred drink"), but each states a different claim.
- A is the **current truth** on that dimension.
- B and C are **historical** — their claims on that dimension are outdated.
- However, B and C may contain OTHER information (on different dimensions) that is still valid and useful.

Example:
- C: "The user lives in Beijing and works as a software engineer."
- B: "The user moved to Shanghai for a new job." (supersedes C on residence — cannot live in both cities simultaneously)
- A: "The user relocated to Tokyo." (supersedes B on residence)

Here, only the residence dimension evolves (one can only live in one city at a time). The fact "works as a software engineer" in C is still valid — it was never contradicted.

Counter-example (NOT a valid chain):
- "The user likes coffee." → "The user also likes tea."
These are NOT contradictory — preferences can coexist. Do NOT supersede. Just ADD independently.

**Key principle**: A chain node ONLY states the new/changed claim on the contested dimension. It does NOT repeat information from older nodes. Old nodes remain visible and retrievable — there is no need to copy their content forward.

When you see `history_versions` on an existing memory, read the full chain to understand the complete history before making decisions.

## New memories to integrate
{new_memories}

## Operation primitives

You have exactly two primitives:

### ADD — create a new memory node
- `op`: `"ADD"`
- `content`: memory text. Third person ("The user ..."). Self-contained.
- `layer`: `"L2_FACT"` or `"L4_IDENTITY"`.
- `supersedes`: list of existing `memory_id`s that this new node marks as no-longer-current-truth on the contested dimension. Default `[]`. Referenced nodes are preserved as historical versions on the chain.
- `supersede_reason`: (required when `supersedes` is non-empty) a short sentence explaining WHAT specifically contradicts, e.g. "User now prefers tea over coffee". This reason is stored and shown to downstream readers.
- `tags`: 1–3 lowercase topic keywords. **Prefer reusing tags from the existing tags list below. Only create a new tag when no existing one fits.**

### DELETE — logically remove an existing memory
- `op`: `"DELETE"`
- `memory_id`: the existing node to remove. Use only when a concurrent ADD has absorbed its content entirely.

## Existing tags (reuse these when possible, avoid synonyms)
{existing_tags}

## Objectives

### Objective 1 — Build contradiction chains for ACTUALLY CONTRADICTORY state
When a new memory and existing memories make claims on the **same dimension** that **cannot both be currently true**, form a chain: emit an ADD whose `supersedes` points at the existing head node.

Key test: "If a reader asks 'what does the user currently do/prefer/have on dimension X?', would the old node give the WRONG answer?" If yes → supersede it.

**STRICT rules for supersedes:**
- ONLY use `supersedes` when there is a genuine factual contradiction (A says X, B says not-X).
- Do NOT supersede for refinement, elaboration, or accumulation — those go to Objective 2.
- Do NOT supersede just because the new memory adds more detail on the same topic.
- The new head node MUST contain ONLY the new/changed claim. Do NOT copy content from the superseded node — it is still visible in the chain.

### Objective 2 — Consolidate fragmented memories
When memories describe the same topic compatibly (no contradiction), merge them:
- Emit one ADD with merged content (`supersedes: []`).
- Emit DELETE for each absorbed existing node.
- The merged content MUST preserve every meaningful fact from all sources.

### Objective 3 — Granularity control
Each memory should represent one coherent thought with complete internal logic. Avoid extremes:
- Too fine-grained: fragmentary sentences that lack context on their own
- Too coarse-grained: a single topic that has grown too large (e.g. exceeding ~2000 characters), packing too many aspects into one node. In this case, split the oversized node into multiple finer-grained aspect nodes — each covering a distinct sub-aspect — while ensuring NO information is lost across the split.

When in doubt, prefer slightly coarser (keep related aspects together) over splitting too aggressively.

### Objective 4 — Do not touch what does not need changing
Memories you choose not to operate on remain untouched. Do not emit ops against them.

## Absolute prerequisite — lossless information preservation

**No information from any source — new or existing — may be lost after your ops execute.** Every meaningful fact must remain retrievable: as content of a chain node, as part of a consolidated ADD, as an independent ADD, or untouched in its original node.

**However, for supersedes chains**: the old node is NOT deleted — it remains in the chain and is visible to readers. Therefore you do NOT need to copy old node content into the new head. Only write the new/changed claim. Information is preserved because old nodes are still there.

## Hard constraints

- Every ADD's `content` is self-contained and third-person. Language MUST match input memories.
- `memory_at: null` means time unknown. Never fabricate chronology; never cite "age" as a reason for superseding.
- Prefer same-layer ops (L6↔L6, L2↔L2). Cross-layer only when semantic meaning genuinely transfers.
- Output language matches input memory language.

## Acceptance self-check (run mentally before emitting)

1. **Lossless**: every fact from every new memory, and every fact from every existing node you are superseding or deleting, ends up somewhere in the final state.
2. **Fragmentation cleared**: no two non-chain nodes on clearly the same topic remain as separate fragments where a merge would improve retrievability.
3. **No no-op DELETEs**: every DELETE has a compensating ADD that absorbs its content.

## Output format

Top-level is a JSON array of groups. Each group is one logical update:

```
{{"reason": "<short description>",
  "ops": [ <op>, <op>, ... ]}}
```

Shape-only reference:

```json
[
  {{
    "reason": "<why this group exists>",
    "ops": [
      {{"op": "ADD", "content": "The user ...", "layer": "L4_IDENTITY",
        "supersedes": [], "tags": ["..."]}},
      {{"op": "ADD", "content": "The user now prefers tea.", "layer": "L4_IDENTITY",
        "supersedes": ["<existing_id>"], "supersede_reason": "Previously preferred coffee, now prefers tea", "tags": ["..."]}},
      {{"op": "DELETE", "memory_id": "<existing_id>"}}
    ]
  }}
]
```

## Output contract (strict)

1. Output MUST be a single JSON array wrapped in a fenced code block that starts with ```json and ends with ```.
2. NO prose, markdown, apologies, or chain-of-thought outside the fenced block.
3. If nothing needs to change, output `[]` inside the fenced block.
4. Field types must match exactly: `supersedes` and `tags` are arrays (use `[]` when empty).

Now produce the JSON array."""

SUMMARY_ZH = """为以下对话内容生成简洁的摘要。

内容:
---
{content}
---

当前时间: {current_time}

要求:
1. **第三人称**: 以"用户..."描述用户——不要使用没有明确先行词的代词。
2. **长度**: 1-3句话，最多200字。
3. **优先级（内容超出长度限制时）**:
   a) 变化、决定、承诺（最高）
   b) 明确的偏好、态度、喜恶
   c) 关键事件和事实
   d) 背景信息（最低）
4. **保留偏好信号**: 保留任何直接或间接表达的喜好、厌恶、态度或观点——即使看起来不重要。
5. **自包含**: 读者应该在不看原始对话的情况下就能理解摘要。
6. **不要编造**: 不要添加原始内容中没有的信息。
7. **语言**: 输出语言必须与输入语言一致（中文输入→中文输出）。
8. **时间**:
   - 如果提供了 `当前时间`，将相对时间表达转换为绝对引用（"上周" → 对应的日期左右）。
   - 如果 `当前时间` 为空，重写句子使其不包含时间（避免在输出中留下"上周"/"昨天"等原始表达）。

## 输出约定

严格格式要求:
1. 只输出摘要文本——一段1-3句话的纯文本。
2. 不要用引号、反引号、代码块或任何 markdown 包裹输出。
3. 不要添加前缀或标签（不要"摘要："、"Summary："、"这是..."等）。
4. 不要添加尾部解释或元评论。
5. 如果内容太简单不值得总结，输出一句描述最显著元素的话——不要输出空字符串。

现在生成摘要。"""

SUMMARY_EN = """Generate a concise summary of the following conversation content.

Content:
---
{content}
---

Current time: {current_time}

Requirements:
1. **Third-person voice**: Describe the user as "The user ..." — do not use pronouns without clear antecedent.
2. **Length**: 1-3 sentences, max 200 words.
3. **Priorities (when content exceeds the length budget)**:
   a) Changes, decisions, commitments (highest)
   b) Explicit preferences, attitudes, dislikes/likes
   c) Key events and facts
   d) Background context (lowest)
4. **Preserve preference signals**: Retain any expression of likes, dislikes, attitudes, or opinions — direct or implied — even if minor.
5. **Self-contained**: A reader should understand the summary without seeing the original conversation.
6. **No fabrication**: Do NOT add information not present in the original content.
7. **Language**: Output language MUST match input language (English → English, Chinese → Chinese).
8. **Time**:
   - If `current_time` is provided, resolve relative expressions to absolute references ("last week" → around the corresponding date).
   - If `current_time` is empty, rewrite sentences atemporally (avoid leaving raw "last week" / "yesterday" in the output).

## Output contract

Strict formatting rules:
1. Output the summary text ONLY — one paragraph of 1-3 sentences, plain prose.
2. Do NOT wrap the output in quotes, backticks, code fences, or any markdown.
3. Do NOT add a prefix or label (no "Summary:", "摘要：", "Here is ...", etc.).
4. Do NOT add trailing explanations or meta-commentary.
5. If the content is too trivial to summarize meaningfully, output a single sentence describing the most salient element — do not output an empty string.

Now produce the summary."""

def pick(zh_prompt: str, en_prompt: str, text: str) -> str:
    return zh_prompt if is_chinese(text) else en_prompt
