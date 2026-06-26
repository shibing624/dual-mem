# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Bilingual prompt templates for the dual-mem agents (gate, extract, reconcile,
summary, system2 ops, cross-domain sweep). Dual ZH/EN forms picked by pick() from input language.
"""
from dual_mem.providers.llm import is_chinese


# =====================================================================
# Attentional Gate — LLM-primary three-dimension scoring with heuristic fallback
# =====================================================================

GATE_CONTEXT_ZH = """对话上下文 (Agent 的上一条消息):
---
{agent_context}
---
注意: 用户的回复可能是对上面这条消息的回应。即使回复很短, 也要结合上下文判断其信息价值。"""

GATE_CONTEXT_EN = """Agent context (the assistant's previous message):
---
{agent_context}
---
Note: the user's reply may respond to the message above. Even a short reply can carry value in context."""

GATE_ZH = """你是一个记忆价值评估系统。判断一段对话内容是否值得被记忆系统长期记住。

输入可能是单条消息，也可能是多轮对话（用户-助手来回多轮）。**请把整段对话作为整体评估**，不要逐条单独评分。

请从以下三个维度对内容进行评分（0.0 ~ 1.0）：

1. **novelty**（新颖度/信息量）
   - 纯寒暄、敷衍回复、无实质内容 → 0.0~0.1（如 "嗯嗯好的"）
   - 多轮对话中只要有一轮包含有价值的新信息，就应给较高分
   - 例："我比较喜欢川菜，但是花生过敏" → 0.85

2. **biographical_relevance**（传记相关性）
   - 涉及用户持久属性（身份、偏好、习惯、家庭、健康、工作、重大事件）→ 高分
   - 安全关键信息（过敏、疾病等）→ 极高分
   - 与用户画像无关的闲聊 → 低分

3. **emotional_arousal**（情绪唤醒度）
   - 强烈情绪表达 → 高分
   - 平淡/无情绪 → 低分
   - 多轮对话中取最高情绪强度

{context_section}

对话内容：
---
{content}
---

只输出严格 JSON，不要其它文字：
{{
    "novelty": 0.0,
    "biographical_relevance": 0.0,
    "emotional_arousal": 0.0,
    "reason": "一句话简要说明判断依据"
}}"""

GATE_EN = """You are a memory value gate. Decide whether a passage is worth long-term memory.

Input may be a single message or a multi-turn dialogue (user-assistant turns). Score the
WHOLE passage as one unit; do NOT score each turn separately.

Rate three dimensions on a 0.0 to 1.0 scale:

1. **novelty** (new-information density)
   - Pure pleasantry / filler / no substantive info → 0.0–0.1 (e.g. "ok thanks")
   - In multi-turn input, if any single turn carries valuable new info, score high overall
   - Example: "I love Sichuan food but I'm allergic to peanuts" → 0.85

2. **biographical_relevance** (relevance to who the user is)
   - Touches a durable attribute (identity, preference, habit, family, health, work, major events) → high
   - Safety-critical info (allergies, illnesses) → very high
   - Pure chit-chat unrelated to the user's profile → low

3. **emotional_arousal** (intensity of emotional expression)
   - Strong emotion → high
   - Flat / neutral → low
   - In multi-turn input, take the maximum intensity across turns

{context_section}

Conversation content:
---
{content}
---

Output strict JSON only, nothing else:
{{
    "novelty": 0.0,
    "biographical_relevance": 0.0,
    "emotional_arousal": 0.0,
    "reason": "one short sentence explaining the rationale"
}}"""


# =====================================================================
# Lightweight Extractor — facts + identity + basic_info + emotion + intentions + ephemeral
# =====================================================================

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

### intentions (L7_INTENTION 候选)
用户表达的**具体未来事件或计划**，要求**明确行动 + 时间边界**（明确或隐含）。

仅当用户表达了**会发生或会过期**的具体未来事件/计划时填入。否则给空数组 `[]`。
✅ "用户正在准备一场工作面试。"
✅ "用户计划下个月去音乐节。"
❌ "用户想成为更好的人。"（无具体行动）
❌ "用户重视家庭。"（人生信念→identity，不是 intention）

### 基本结构化属性——放入 basic_info，不要作为 identity/facts 输出
稳定的结构化个人属性——**姓名、年龄、所在地、职业、雇主**——不作为 identity 或 fact 记忆输出，而是放入输出 JSON 的 `basic_info` 对象，只填对话中明确提到或更新的字段（键名用 name/age/location/occupation/employer），未提及的字段不要出现。若没有任何基本属性，`basic_info` 输出空对象 `{{}}`。

- 不要生成仅复述这五个属性的 identity 记忆。
- 例外: 如果属性附带丰富的叙事上下文（如"用户2023年从工程师转为产品经理，因为想更贴近用户"），该叙事属于 `identity`；基本事实仍放入 `basic_info`。

### 跨层重叠记忆
一段对话可能同时产出多条 facts + identity 记忆，内容可能有重叠——这是正常的。

### 粒度规则（全局默认）

- 每条 memory 应是一个**可独立检索**的单元：一个事实、事件、计划或偏好**状态**。
- **仅**当多个句子表达**同一时刻、同一语义**的同一信息（换说法）时才合并为一条。
- **必须拆分**当：
  - 用户的偏好、观点或处境**发生变化**（以前 vs 现在、不再、改成、instead、but now）；
  - 不同的事件、日期、地点或实体；
  - 同一主题的不同方面（如饮食偏好 vs 烹饪习惯）。
- 不要把一次偏好演变总结成一条长 narrative；下游 reconcile 会处理演化链。
- **facts (L2)**：短句、原子化陈述，每个可检索 fact 一条。
- **identity (L4)**：每条只表达一个态度/偏好；演变中的偏好拆成多条，而非一条概括整个演变过程。

**示例 — 偏好变化应拆分**：
对话："我以前更喜欢喝茶，但最近改喝手冲咖啡了，周末还会去咖啡馆。"
✅ identity: "用户以前更喜欢喝茶。"
✅ identity: "用户现在更偏好手冲咖啡。"
✅ facts: "用户最近开始喝手冲咖啡，周末会去咖啡馆。"
❌ 不要合并为一条："用户以前喜欢茶但现在改喝咖啡并周末去咖啡馆。"

## emotion 与 is_ephemeral

整段对话给出整体情绪：
- `valence`: -1.0 ~ 1.0 之间（负面到正面），无明显情绪 → 0.0
- `arousal`: 0.0 ~ 1.0 之间（平淡到强烈），无明显情绪 → 0.0
- `dominant_emotion`: 主要情绪标签（如"焦虑"、"开心"、"平静"），无则 null

如果整段对话都是纯寒暄/无信息量（如只有"好的"、"嗯嗯"），`is_ephemeral=true`，并把
`identity` 与 `facts` 都置空数组；只要有任何实质信息就 `is_ephemeral=false`。

## speculate 字段（适用于 identity 和 facts）

`speculate` 是一个可选的分析注释，用于模糊或间接信号的情况。当背后的动机、真实原因或隐含偏好在用户的话语中不明确时填写；否则为 null。

- **对于 identity**: 在态度信号模糊时解读隐含态度。
- **对于 facts**: 当真实原因不明确时解释事件为何发生。
- 不要将 speculate 内容复制到 `content` 字段中。
- 对于清晰明确的陈述不要编造 speculate。

## Tags

每条 identity 和 fact 记忆必须包含 `tags` 字段——1 到 3 个小写主题关键词（如 "音乐"、"旅行"、"工作"、"美食"、"健康"、"社交"、"技术"）。

## 输出格式

只输出 JSON 对象，不要其它文字：

```json
{{
  "is_ephemeral": false,
  "emotion": {{"valence": 0.0, "arousal": 0.0, "dominant_emotion": null}},
  "identity": [
    {{"content": "用户...", "speculate": null, "tags": ["..."]}}
  ],
  "facts": [
    {{"content": "用户...", "speculate": null, "tags": ["..."]}}
  ],
  "intentions": [
    {{"content": "用户...", "trigger_time_description": null, "tags": ["..."]}}
  ],
  "basic_info": {{}}
}}
```"""

EXTRACT_GATE_APPEND_ZH = """

## gate_decision（与提取同一次输出）

同时评估该对话是否值得长期记忆（三个维度 0.0–1.0）：
- `novelty`：信息新颖度
- `biographical_relevance`：与用户传记/偏好的相关性
- `emotional_arousal`：情绪唤醒度
- `reason`：简短理由

在 JSON 顶层增加 `gate_decision` 对象（与 identity/facts 同级）：
```json
"gate_decision": {{"novelty": 0.8, "biographical_relevance": 0.7, "emotional_arousal": 0.2, "reason": "..."}}
```"""

EXTRACT_GATE_APPEND_EN = """

## gate_decision (same response as extraction)

Also score whether this passage is worth long-term memory (each 0.0–1.0):
- `novelty`, `biographical_relevance`, `emotional_arousal`, `reason`

Add a top-level `gate_decision` object alongside identity/facts:
```json
"gate_decision": {{"novelty": 0.8, "biographical_relevance": 0.7, "emotional_arousal": 0.2, "reason": "..."}}
```"""

EXTRACT_RETRY_APPEND_ZH = """

## 重要：输出格式（重试）

上一次输出无法解析为 JSON。这次**只输出一个 JSON 对象**，不要任何前后说明、不要 Markdown 代码块标记、不要多余文字。第一个字符必须是 `{{`，最后一个字符必须是 `}}`。"""

EXTRACT_RETRY_APPEND_EN = """

## IMPORTANT: output format (retry)

The previous output could not be parsed as JSON. This time output **exactly one JSON object** with no prose before/after, no Markdown code fences. The first character must be `{{` and the last `}}`."""

EXTRACT_EN = """You are an expert memory analyst. Extract structured user information from the conversation below.

Conversation content:
---
{content}
---

Current time: {current_time}

## Core Principles

1. **User as subject**: Every memory describes the user in third person. Use "The user ...". Each memory must be a complete, self-contained statement — never omit circumstances, conditions, or reasons that give the memory meaning.
2. **Preserve details**: Retain ALL specific information (place names, person names, brands, numbers, etc.). Do NOT generalize or simplify.
3. **Language consistency**: Output language MUST match the input language.
4. **Time handling**:
   - If the conversation references a time:
     - Time-sensitive content: embed the time naturally; resolve relative expressions using `current_time` when provided.
     - Not time-sensitive: keep `content` atemporal.
   - If `current_time` is empty AND content has a relative expression: rewrite `content` without the raw relative expression.

## Memory Layers

### identity (L4_IDENTITY)
User's personal attributes, preferences, attitudes, opinions, values, and feelings — what makes the user unique as a person.

Watch for **implicit preference signals** (resuming, enthusiasm, burden words, loss-of-interest, "but now ...", re-engagement) and emit explicit identity memories that state the attitude.

### facts (L2_FACT)
Objective events, experiences, and factual occurrences — what happened or is planned, NOT how the user feels about it.

### intentions (L7_INTENTION candidates)
A CONCRETE future event or plan with clear action + temporal boundedness (explicit or implicit).
GOOD: "The user is preparing for a job interview."
BAD: "The user wants to be a better person." (no concrete action — belongs to identity, not intention.)

### Basic structured attributes — go into `basic_info`, NOT identity/facts
Stable attributes (name, age, location, occupation, employer) go into the `basic_info` object only, with the keys present only when mentioned. Empty `{{}}` if none.

Exception: a basic attribute with rich narrative context (e.g. "The user became a product manager in 2023 after 5 years as an engineer") goes into `identity`; the bare fact still goes into `basic_info`.

### Granularity (default for all conversations)

- Emit ONE memory per distinct, retrievable unit: a fact, event, plan, or preference **state**.
- Merge ONLY when multiple sentences express the **same** information at the **same** time (rephrasing).
- ALWAYS split when:
  - the user's preference, opinion, or situation **changes** (before vs after; no longer; instead; but now);
  - different events, dates, places, or entities;
  - different aspects of the same topic (e.g. food preference vs cooking habit).
- Do NOT bundle a preference arc into one long narrative; the reconciler builds evolution chains downstream.
- **facts (L2)**: short, atomic statements — one retrievable fact per item.
- **identity (L4)**: one attitude/preference per item; split evolving preferences across items.

**Example — preference change should split**:
Dialogue: "I used to prefer tea, but lately I've switched to pour-over coffee and visit cafés on weekends."
✅ identity: "The user used to prefer tea."
✅ identity: "The user now prefers pour-over coffee."
✅ facts: "The user recently started drinking pour-over coffee and visits cafés on weekends."
❌ Do NOT merge into one item covering the whole arc.

## emotion and is_ephemeral

Score the WHOLE passage:
- `valence`: -1.0 to 1.0 (negative to positive), 0.0 when neutral
- `arousal`: 0.0 to 1.0 (flat to intense), 0.0 when neutral
- `dominant_emotion`: short label (e.g., "anxious", "excited"), null when none

If the whole passage is pure pleasantry / no substantive info (e.g. just "ok"/"thanks"), set `is_ephemeral=true` and leave `identity` / `facts` as empty arrays.

## speculate field (identity AND facts)

Optional analytical note for ambiguous signals; null when the statement is explicit. Do not duplicate speculate content into `content`.

## Tags

Each identity / fact memory has 1-3 lowercase topic keywords.

## Output format

Output ONLY the JSON object, no extra text:

```json
{{
  "is_ephemeral": false,
  "emotion": {{"valence": 0.0, "arousal": 0.0, "dominant_emotion": null}},
  "identity": [
    {{"content": "The user ...", "speculate": null, "tags": ["..."]}}
  ],
  "facts": [
    {{"content": "The user ...", "speculate": null, "tags": ["..."]}}
  ],
  "intentions": [
    {{"content": "The user ...", "trigger_time_description": null, "tags": ["..."]}}
  ],
  "basic_info": {{}}
}}
```"""


# =====================================================================
# Search-query expansion (kept as-is from original dual_mem; off by default)
# =====================================================================

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

SEARCH_QUERY_EN = """You are a search query generator. Given a list of newly extracted memories, generate a set of short search queries to find related existing memories in a vector database.

The goal is to maximize recall — find existing memories semantically related to the new memories, even if wording differs.

## New memories:
{new_memories}

## Guidelines

Cover:
- Key topics, entities, and themes
- Rephrased / abstracted versions of core concepts
- Likely related concepts in the user's memory store

Output a JSON array of query strings (5-15 short focused queries):

["query1", "query2", "query3", ...]

Output JSON array only, no other text."""


# =====================================================================
# Reconciler — integrate new memories into existing chain (kept, with sharper rules)
# =====================================================================

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

这里只有居住地维度在演进。C 中"从事软件工程师工作"的事实仍然有效——它从未被否定。

## 待整合的新记忆
{new_memories}

## 操作原语

你有两种操作原语:

### ADD — 创建一个新记忆节点
- `op`: `"ADD"`
- `content`: 记忆文本。第三人称（"用户..."）。自包含。
- `layer`: `"L2_FACT"` 或 `"L4_IDENTITY"`。
- `supersedes`: 此新节点在争议维度上取代的已有 `memory_id` 列表。默认 `[]`。被引用的节点作为历史版本保留在链上。
- `tags`: 1-3 个小写主题关键词。**优先从下方已有标签列表中选取，仅在现有标签都不合适时才新增。**
- `update_type`: 该新节点与已有记忆的关系类型（必填，五选一）：
  - `"OVERRIDE"`: 同维度状态变化，新节点取代旧节点（必须配合 `supersedes`）。
  - `"SUPPLEMENT"`: 与已有记忆兼容、独立累加（`supersedes: []`）。
  - `"TEMPORAL"`: 临时/短期变化（"今天想吃...", "最近在..."），可选 `temporal_scope` 描述有效范围。
  - `"NEGATE"`: 显式否定旧主张（"不再喜欢...", "已经离开..."），必须配合 `supersedes` 指向被否定的节点。
  - `"CONFLICT"`: 矛盾但无法判断真伪，保留两条共存（`supersedes: []`），由后续读侧消歧。
- `temporal_scope`: 仅在 `update_type=="TEMPORAL"` 时填写，简短描述有效范围（如"今天"、"本周"、"本次出差期间"）。其它情况省略或填 `null`。

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
每条记忆应代表一个连贯的想法。避免两个极端:
- 过于细粒度: 缺乏独立上下文的碎片化句子
- 过于粗粒度: 同一主题膨胀过大（>2000 字符），应拆分为多条更细维度的节点

犹豫时，倾向于稍粗一些。

### 目标 4 — 不要动不需要改变的东西
你选择不操作的记忆保持不变。

## 绝对前提——信息无损保留

**任何来源——新的或已有的——的任何信息在你的操作执行后都不得丢失。** 每条有意义的事实都必须保持可检索: 作为链节点的内容、作为合并 ADD 的一部分、作为独立 ADD、或在原始节点中保持不变。

## 硬约束

- 每个 ADD 的 `content` 必须自包含且为第三人称。语言必须与输入记忆一致。
- `memory_at: null` 表示时间未知。绝不要编造时间线；绝不要以"年代久远"为由取代记忆。
- 优先同层操作（L4↔L4，L2↔L2）。跨层仅在语义意义确实转移时使用。

## 输出格式

输出一个 JSON 对象，顶层键 `updates` 是一个分组数组。每组是一个逻辑更新:

```json
{{
  "updates": [
    {{
      "reason": "<为什么需要这组操作>",
      "ops": [
        {{"op": "ADD", "content": "用户...", "layer": "L4_IDENTITY",
          "supersedes": [], "tags": ["..."], "update_type": "SUPPLEMENT"}},
        {{"op": "ADD", "content": "用户...", "layer": "L4_IDENTITY",
          "supersedes": ["<existing_id>"], "tags": ["..."], "update_type": "OVERRIDE"}},
        {{"op": "DELETE", "memory_id": "<existing_id>"}}
      ]
    }}
  ]
}}
```

## 输出约定（严格）

1. 输出必须是一个有效的 JSON 对象，顶层键为 `updates`（一个数组）。
2. 不要有任何文字、markdown、道歉或思维链。
3. 如果不需要任何变更，输出 `{{"updates": []}}`。
4. 字段类型必须严格匹配: `supersedes` 和 `tags` 是数组（为空时用 `[]`）。

现在输出 JSON。"""

# conservative 策略附加硬规则：禁止跨节点合并，可数事件独立保留 —— 拼接到 RECONCILE_ZH 之后。
RECONCILE_POLICY_CONSERVATIVE_ZH = """

## 保守策略（强制覆盖以上目标 2）

本次运行为**高召回保守模式**，必须遵守：
- **禁止合并**：不要把多条记忆合并成一条，不要用 DELETE 去吸收/合并兼容的记忆。
- **可数事件独立**：不同实体、不同日期、不同对象的事件（如不同人生娃、不同次购买、不同次到访），
  即使主题相同，也必须各自独立保留为 `update_type="SUPPLEMENT"`、`supersedes: []` 的 ADD，绝不合并计数。
- **仅状态变化才 supersede**：只有同一维度的状态发生真实变化（OVERRIDE / NEGATE）时才用 `supersedes`。
- **DELETE 仅限精确重复**：只有内容几乎完全重复时才 DELETE，禁止用 DELETE 做语义归并。
- 拿不准时一律 `SUPPLEMENT` 独立保留，宁可冗余也不要丢证据。"""

RECONCILE_POLICY_CONSERVATIVE_EN = """

## Conservative policy (OVERRIDES Objective 2 above)

This run is in HIGH-RECALL conservative mode. You MUST:
- **No merging**: never fuse multiple memories into one; never use DELETE to absorb/merge compatible memories.
- **Keep countable events separate**: distinct entities / dates / objects (different people's births, different
  purchases, different visits) MUST each stay as an independent ADD with `update_type="SUPPLEMENT"` and
  `supersedes: []`, even on the same topic — never merge them into a single count.
- **Supersede only on state change**: use `supersedes` ONLY for a real same-dimension state change (OVERRIDE / NEGATE).
- **DELETE only for exact duplicates**: DELETE is allowed only for near-identical content, never for semantic merging.
- When unsure, default to an independent `SUPPLEMENT` — prefer redundancy over losing evidence."""

RECONCILE_EN = """You are a memory management system. Integrate a batch of new memories into the existing memory base while keeping it clean, retrievable, and losslessly informative.

Current time: {current_time}
(`current_time` is for reference only. Age alone is NOT a reason to supersede a memory.)

## Existing memories
{existing_memories}

Fields:
- `memory_id`: stable id (only on chain heads — these are the only IDs you may reference).
- `content`: stored text.
- `memory_at`: timestamp (ISO). May be `null`.
- `layer`: `l2_fact` or `l4_identity`.
- `tags`: topic keywords.

Memories may include `history_versions` — that's the supersedes chain.

## Understanding supersedes chains

A chain represents **state evolution on one specific dimension**.

Given chain A → B → C (A latest, C oldest):
- All three describe the SAME dimension (e.g. "preferred drink") but state different claims.
- A is current truth on that dimension; B and C are historical claims.
- B/C may carry OTHER information on different dimensions that is still valid.

Example:
- C: "The user lives in Beijing and works as a software engineer."
- B: "The user moved to Shanghai for a new job." (supersedes C on residence)
- A: "The user relocated to Tokyo." (supersedes B on residence)

Counter-example (NOT a chain):
- "The user likes coffee." → "The user also likes tea." (preferences can coexist; ADD independently.)

## New memories to integrate
{new_memories}

## Operation primitives

### ADD — create a new memory node
- `op`: `"ADD"`
- `content`: third-person, self-contained.
- `layer`: `"L2_FACT"` or `"L4_IDENTITY"`.
- `supersedes`: list of existing `memory_id`s that this new node marks as no-longer-current-truth on the contested dimension. Default `[]`.
- `tags`: 1-3 lowercase keywords; prefer reusing the existing tags list below.
- `update_type` (REQUIRED): the relationship between this new node and existing memories. One of:
  - `"OVERRIDE"`: same-dimension state change; new replaces old (REQUIRES non-empty `supersedes`).
  - `"SUPPLEMENT"`: compatible, independent accumulation (`supersedes: []`).
  - `"TEMPORAL"`: short-lived / temporary change ("today wants ...", "currently into ..."); set `temporal_scope`.
  - `"NEGATE"`: explicit negation of an old claim ("no longer likes ...", "has left ..."); REQUIRES `supersedes`.
  - `"CONFLICT"`: contradictory but unable to judge; keep both for read-side disambiguation (`supersedes: []`).
- `temporal_scope`: only when `update_type=="TEMPORAL"`; short text (e.g. "today", "this week", "during current trip"). Otherwise omit or `null`.

### DELETE — logically remove an existing memory
- `op`: `"DELETE"`
- `memory_id`: existing node to soft-delete. Use only when a concurrent ADD has absorbed its content.

## Existing tags (reuse when possible)
{existing_tags}

## Objectives

### 1 — Build supersedes chains for ACTUAL contradictions only
Use `supersedes` only when claims on the SAME dimension cannot both be currently true. NOT for refinement / accumulation. The new head node must contain ONLY the new/changed claim — do NOT copy old node content forward.

### 2 — Consolidate fragmented memories
When memories describe the same topic compatibly: emit one merged ADD (`supersedes: []`) plus DELETE per absorbed node. Preserve every meaningful fact from all sources.

### 3 — Granularity control
One memory = one coherent thought. Avoid both fragmentation and over-bundling (>2000 chars). When in doubt, prefer slightly coarser.

### 4 — Do not touch what does not need changing
Untouched memories remain untouched.

## Absolute prerequisite — lossless preservation

No information from any source may be lost after your ops execute.

## Hard constraints

- ADD `content` is self-contained and third-person; language matches input.
- `memory_at: null` means time unknown — never fabricate chronology, never cite "age" as a reason.
- Prefer same-layer ops; cross-layer only when meaning genuinely transfers.

## Output format

```json
{{
  "updates": [
    {{
      "reason": "<why this group exists>",
      "ops": [
        {{"op": "ADD", "content": "The user ...", "layer": "L4_IDENTITY",
          "supersedes": [], "tags": ["..."], "update_type": "SUPPLEMENT"}},
        {{"op": "ADD", "content": "The user now prefers tea.", "layer": "L4_IDENTITY",
          "supersedes": ["<existing_id>"], "tags": ["..."], "update_type": "OVERRIDE"}},
        {{"op": "DELETE", "memory_id": "<existing_id>"}}
      ]
    }}
  ]
}}
```

## Output contract (strict)

1. Valid JSON with top-level key `updates` (array).
2. No prose, markdown, apologies, or chain-of-thought.
3. Empty case: `{{"updates": []}}`.
4. `supersedes` and `tags` are arrays (use `[]` when empty).

Output JSON now."""


# =====================================================================
# Summarizer — long-content summary (>=500 chars)
# =====================================================================

SUMMARY_ZH = """为以下对话内容生成简洁的摘要。

内容:
---
{content}
---

当前时间: {current_time}

要求:
1. **第三人称**: 以"用户..."描述用户——不要使用没有明确先行词的代词。
2. **长度**: 1-3 句话，最多 200 字。
3. **优先级（内容超出长度限制时）**:
   a) 变化、决定、承诺（最高）
   b) 明确的偏好、态度、喜恶
   c) 关键事件和事实
   d) 背景信息（最低）
4. **保留偏好信号**: 保留任何直接或间接表达的喜好、厌恶、态度或观点。
5. **自包含**: 读者应该在不看原始对话的情况下就能理解摘要。
6. **不要编造**: 不要添加原始内容中没有的信息。
7. **语言**: 输出语言必须与输入语言一致。
8. **时间**:
   - 如果提供了 `当前时间`，将相对时间表达转换为绝对引用。
   - 如果 `当前时间` 为空，重写句子使其不包含时间。

## 输出约定

1. 只输出摘要文本——一段 1-3 句话的纯文本。
2. 不要用引号、反引号、代码块或任何 markdown 包裹输出。
3. 不要添加前缀或标签（不要"摘要："、"Summary："、"这是..."等）。
4. 不要添加尾部解释或元评论。

现在生成摘要。"""

SUMMARY_EN = """Generate a concise summary of the following conversation content.

Content:
---
{content}
---

Current time: {current_time}

Requirements:
1. **Third-person voice**: Describe the user as "The user ...".
2. **Length**: 1-3 sentences, max 200 words.
3. **Priorities (when content exceeds the length budget)**:
   a) Changes, decisions, commitments (highest)
   b) Explicit preferences, attitudes, dislikes/likes
   c) Key events and facts
   d) Background context (lowest)
4. **Preserve preference signals**.
5. **Self-contained**.
6. **No fabrication**.
7. **Language**: must match the input.
8. **Time**:
   - With `current_time`, resolve relative expressions to absolute references.
   - Without it, rewrite atemporally.

## Output contract

1. Output the summary text ONLY — one paragraph of 1-3 sentences, plain prose.
2. Do NOT wrap in quotes, backticks, code fences, or markdown.
3. Do NOT add a prefix or label.
4. Do NOT add trailing explanations or meta-commentary.

Output the summary now."""


# =====================================================================
# System2 Agent — schema/intention ops emission
# =====================================================================

SYSTEM2_OPS_ZH = """你是一个认知加工 Agent，负责从用户的事实聚类中演化高层认知结构（L6 Schema / L7 Intention），并以一组操作（ops）的形式输出。

## L6 Schema 是什么？

Schema 捕获用户在**特定领域**内的**一个**行为模式，包含三个要素：
- **场景（Circumstance）**: 该模式发生的领域/话题/场景。Schema 必须限定在其场景内，不要跨域泛化。
- **模式（Pattern）**: 用户在该场景下的惯常行为、思维方式或行动倾向。
- **洞察（Insight）**: 底层心理驱动力或心智模型。

### 内容格式
用一句话组合三要素：
"当[场景]时，用户[模式]——反映了[洞察]。"

### 规则
- **原子化**: 一个 Schema 只包含一个模式。两个不同模式 → 两个 Schema。
- **不可变**: Schema 创建后内容永远不变。不要修改已有 Schema 的内容。
- **累积证据**: 新事实支持已有 Schema → 调 `add_evidence` 添加证据，不要重新创建。
- **域内约束**: 不要跨域。跨域抽象由系统的其他流程处理。

✅ "当做饭时，用户严格按菜谱步骤精确称量——反映了用外部结构管理不确定性的需要。"
❌ "用户对很多事充满热情且追求品质。"（无场景、太泛）

## L7 Intention 是什么？

Intention 是用户表达的**具体未来事件或计划**。必须有：
- **明确行动**: 用户将要做的事（不是感受或信念）
- **时间边界**: 可以明确（"下周"）或隐含（"正在准备"）

### 规则
- 没有具体行动或事件 → 不是 Intention，考虑归入 Schema
- 人生愿景、价值观 → Schema，不是 Intention
- 一个事件一个 Intention，保持简洁

## 工作流程

对于每个事实聚类：
1. **搜索已有 Schema**: 如果已有 Schema 已覆盖了该模式，调 `add_evidence` 链接新事实——不要重新创建。
2. **创建新 Schema**: 仅当没有已有 Schema 覆盖时。一句话，三要素。
3. **检测 Intention**: 事实中包含具体未来计划/事件 → 创建 L7 Intention。
4. **建立关系**: 两个 Schema 主题相关 → 用 `add_edge`（RELATED_TO）。

## 输出格式

只输出一个 JSON 对象，键 `ops` 是操作数组，不要任何解释或代码块外文字。每个 op 是以下四类之一：
{{"ops": [
  {{"op": "create_schema", "content": "当...时，用户...——反映...", "tags": ["..."], "evidence": ["fact_id", ...]}},
  {{"op": "create_intention", "content": "...", "tags": ["..."], "evidence": ["fact_id", ...]}},
  {{"op": "add_evidence", "schema_id": "已有schema_id", "evidence": ["fact_id", ...]}},
  {{"op": "add_edge", "from_id": "...", "to_id": "...", "rel": "RELATED_TO"}}
]}}

## 原则
- Schema 内容创建后不可变——绝不重新创建已存在的 Schema
- 优先 `add_evidence` 而非创建重复节点
- 宁可不创建，也不要创建低质量节点
- 一个 Schema = 一个域内原子模式
- 一个 Intention = 一个具体未来事件
- 打标签时优先使用已有标签列表中的标签

打标签时优先复用已有标签列表中的标签。证据 fact_id 必须来自聚类中给出的 id。
若数据不足以得出可靠结论，输出 {{"ops": []}}。"""

SYSTEM2_OPS_EN = """You are a cognitive processing Agent. Evolve higher-order cognitive structures (L6 Schema / L7 Intention) from the user's fact clusters, and output them as a list of operations (ops).

## What is an L6 Schema?

A Schema captures ONE behavioral pattern in a SPECIFIC domain. Three components:

- **Circumstance**: the domain/topic/situation where this pattern is observed. A Schema MUST stay within its circumstance — do NOT generalize across domains.
- **Pattern**: the user's habitual behavior, thinking style, or action tendency in this circumstance.
- **Insight**: the underlying psychological driver or mental model.

### Content format
Single sentence combining all three:
"When [circumstance], the user [pattern] — reflecting [insight]."

### Rules
- **Atomic**: One pattern per Schema. Two distinct patterns → two Schemas.
- **Immutable**: Once created, content NEVER changes.
- **Evidence only**: New facts support existing Schema → call `add_evidence`. Do NOT recreate.
- **Domain-bound**: Stay within the observed domain. Cross-domain abstraction is handled separately.

✅ "When cooking, the user strictly follows recipe steps and precisely measures ingredients — reflecting a need for external structure to manage uncertainty."
❌ "The user is passionate about many things and values quality." (no circumstance, too vague)

## What is an L7 Intention?

A CONCRETE future event or plan. Requires:
- **A clear action**: something the user will DO
- **Temporal boundedness**: explicit ("next week") or implicit ("currently preparing for")

### Rules
- No concrete action → NOT an Intention. Consider Schema instead.
- Life aspirations, values, visions → Schema, not Intention.
- One event per Intention.

## Workflow

For each cluster of facts:
1. **Search existing Schemas**: if covered → `add_evidence`, do NOT recreate.
2. **Create new Schema**: only when no existing Schema covers this pattern. One sentence, three components.
3. **Detect Intentions**: concrete future plan → create L7 Intention.
4. **Build relationships**: two Schemas thematically related → `add_edge` (RELATED_TO).

## Output format

Output ONLY a JSON object whose key `ops` is the operations array. No prose, nothing outside it. Each op is one of four kinds:
{{"ops": [
  {{"op": "create_schema", "content": "When ..., the user ... — reflecting ...", "tags": ["..."], "evidence": ["fact_id", ...]}},
  {{"op": "create_intention", "content": "...", "tags": ["..."], "evidence": ["fact_id", ...]}},
  {{"op": "add_evidence", "schema_id": "existing_schema_id", "evidence": ["fact_id", ...]}},
  {{"op": "add_edge", "from_id": "...", "to_id": "...", "rel": "RELATED_TO"}}
]}}

## Principles
- Schema content is IMMUTABLE — never recreate what already exists
- Prefer `add_evidence` over duplicates
- Prefer not creating over creating low-quality nodes
- One Schema = one atomic pattern in one domain
- One Intention = one concrete future event
- Reuse tags from the existing tags list when possible

Evidence fact_ids MUST come from the cluster ids provided.
If the data is insufficient for reliable conclusions, output {{"ops": []}}."""


# =====================================================================
# Cross-Domain Sweeper — behavior abstraction + higher-order induction
# =====================================================================

BEHAVIOR_ABSTRACTION_ZH = """任务：你是一名行为心理学家。请把下列 schema 抽象为一段纯心理/行为描述。剥离所有领域具体名词和场景——只输出底层行为风格和心理动机。

语言规则：输出语言必须与输入相同。

输入 schema：
{content}

输出（严格 JSON）：
{{"abstraction_for_embedding": "...纯行为描述，与输入语言一致..."}}"""

BEHAVIOR_ABSTRACTION_EN = """Task: You are a behavioral psychologist. Abstract the following schema into a pure psychological/behavioral description. Strip ALL domain-specific nouns and scenarios — output ONLY the underlying behavioral style and psychological motivation.

LANGUAGE RULE: Your output MUST be in the SAME language as the input schema.

Input schema:
{content}

Output (strict JSON):
{{"abstraction_for_embedding": "...pure behavioral description, same language as input..."}}"""


CROSS_DOMAIN_INDUCTION_ZH = """任务：你是一名深层模式分析师。系统在用户不同生活领域的行为 schema 间发现了结构性共鸣：表面看不相关，但底层行为逻辑高度相似。

请综合出一个**更高阶**的核心模式，解释这些行为为何共现（用户自己未必察觉）：
- 深层认知风格（如何处理信息和做决策）
- 核心心理需求（在最根本层面驱动他们的是什么）
- 隐含的心智模型（这些行为背后的潜在信念体系）

要有洞察力。要精准。质量优于数量——只在连接真正成立、逻辑严密时才输出。证据薄弱或连接表面化时，输出 null。

语言规则：输出（core_pattern, reasoning）必须与输入 schema 相同语言。

输入 schemas：
{patterns}

输出（严格 JSON，或 null 如果没有令人信服的综合）：
{{"core_pattern": "...一句话描述更高阶模式，与输入语言一致...", "reasoning": "...为什么这些 schema 在深层相互连接...", "confidence": 0.85, "schema_ids": ["参与归纳的基础 schema_id", ...]}}"""

CROSS_DOMAIN_INDUCTION_EN = """Task: You are a deep pattern analyst. The system has detected structural resonance among the following schemas from different areas of the user's life. On the surface they appear unrelated, but their underlying behavioral logic is strikingly similar.

Your job: synthesize a HIGHER-ORDER pattern that explains WHY these behaviors co-occur — something the user themselves may not be consciously aware of. Think in terms of:
- Deep cognitive style (how they process information and make decisions)
- Core psychological need (what drives them at a fundamental level)
- Hidden mental model (the implicit belief system behind these behaviors)

Be insightful. Be precise. Quality over quantity — only output a synthesis if the connection is genuinely compelling and logically airtight. If the evidence is weak or the connection is superficial, output null.

LANGUAGE RULE: Your output (core_pattern, reasoning) MUST be in the SAME language as the input schemas below.

Input schemas:
{patterns}

Output (strict JSON, or null if no compelling synthesis):
{{"core_pattern": "...one sentence describing the higher-order pattern, same language as input...", "reasoning": "...why these schemas are connected at a deep level, same language as input...", "confidence": 0.85, "schema_ids": ["basic schema_id involved", ...]}}"""


def pick(zh_prompt: str, en_prompt: str, text: str) -> str:
    """Choose the Chinese or English prompt variant based on the input text's language."""
    return zh_prompt if is_chinese(text) else en_prompt
