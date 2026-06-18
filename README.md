# dual-mem

> 分层记忆内核 —— 从零复现 [Hy-Memory](https://github.com/) 的演化链 / System1-System2 / 8 层记忆框架，封装成一套干净、可测、契约对齐的 Python 记忆 SDK。

`dual_mem` 是一个面向 LLM 应用 / Agent 的**长期记忆**组件。它把零散的对话沉淀成结构化、可演化、可检索的记忆，并通过 **SDK / REST / MCP / CLI / Skill** 五种方式对外提供能力。

核心能力：

- **演化链**：偏好/事实发生变化时（如「最爱 Java」→「转向 Python」、「住上海」→「搬北京」），写侧显式建立 `supersedes`/`superseded_by` 双向指针；旧版本软删保留，检索命中链头时自动展开整条历史（最新→最旧）。
- **System1（写侧同步认知）**：一次写入内集中所有 LLM 调用 —— 抽取 identity/facts、更新 L0 结构化画像、生成摘要、协调演化链。
- **System2（ultra 异步沉淀）**：把同领域事实聚类，升维为高阶认知结构（L6 行为 Schema / L7 意图），并支持跨域升维出「核心 Schema」。
- **三路召回（零 LLM）**：profile（画像/身份/模式）/ proactive（推断意图）/ normal（普通事实知识），叠加演化链展开、BM25+RRF 重排、中文时间词解析、tag 桥接。

---

## 安装

需要 Python ≥ 3.11。

```bash
git clone <repo-url> dual-mem
cd dual-mem
pip install -e .          # 运行时依赖
pip install -e ".[dev]"   # 含测试依赖（pytest / pytest-asyncio / respx）
```

主要依赖：`pydantic` / `pydantic-settings`（配置），`openai`（LLM/Embed 兼容协议），`chromadb`（向量库），`kuzu`（图库），`scikit-learn`+`numpy`（System2 聚类），`rank-bm25`+`jieba`（中文重排），`fastapi`+`uvicorn`（REST），`typer`（CLI），`mcp`（MCP server）。

---

## 快速开始（SDK）

`MemoryClient` 是全异步门面。最小例子：

```python
import asyncio
from dual_mem import MemoryClient


async def main():
    client = MemoryClient(mode="pro", storage_dir="./.dual_mem_data")

    await client.add(
        content="我最爱的编程语言是 Java，已经用了5年。",
        app_id="my_app",
        user_id="alice",
    )

    res = await client.search(
        query="用户的编程语言偏好",
        app_ids=["my_app"],
        user_id="alice",
    )
    # res["memories"] = {"profile": [...], "proactive": [...], "normal": [...]}
    for m in res["memories"]["profile"]:
        print(m["content"], m.get("evolution_chain"))


asyncio.run(main())
```

### 配置（YAML）

配置主源是 YAML 文件，默认读 `~/.dual_mem/config.yaml`（可用环境变量 `DUAL_MEM_CONFIG_FILE` 指定其它路径）。复制仓库根的 `config.example.yaml` 即可：

```bash
mkdir -p ~/.dual_mem
cp config.example.yaml ~/.dual_mem/config.yaml
```

```yaml
mode: pro                       # lite | pro | ultra
storage_dir: ./.dual_mem_data
llm_base_url: https://api.openai.com/v1
llm_api_key: sk-your-llm-key
llm_model: gpt-4o-mini
embed_base_url: https://api.openai.com/v1
embed_api_key: sk-your-embed-key
embed_model: text-embedding-3-small
embed_dim: 1536
auth_disabled: true             # REST 本地鉴权开关
app_whitelist:                  # REST app_id 白名单
  - default
system2_trigger_mode: per_write # per_write | manual | scheduled
```

优先级：显式传参 `MemoryClient(mode=...)` > `DUAL_MEM_` 前缀环境变量（临时覆盖）> YAML 文件 > 默认值。

LLM/Embed 走 OpenAI 兼容协议，`base_url` 可指向任意兼容服务（OpenAI / Hunyuan / 本地推理）。

---

## 三档模式

`mode` 自动派生 `agent_mode` 与 `enable_graph`，决定写入深度、LLM 开销与是否启用图库：

| 维度 | lite | pro | ultra |
|---|---|---|---|
| 一次 `add` 的 LLM 次数 | **0**（仅 embed） | 2~3 | pro + 异步 1~N |
| 写入层 | 仅 L1_RAW | L0 / L1(SHADOW) / L2 / L3 / L4 | + L5 / L6 / L7（图库） |
| System2 异步沉淀 | 否 | 否 | **是**（队列 + 跨域 sweeper） |
| 图库（Kuzu） | 关 | 关 | **开** |
| Writer | MemoryWriter | MemoryWriter | System2Writer（内部仍跑 System1） |
| 适用 | 纯检索 / 离线写入 | 个性化助手 | 深度画像 / 认知沉淀 |

> lite 与 pro 共用 `MemoryWriter`，差别仅在 `agent_mode`；ultra 用 `System2Writer`，写完 System1 后 fire-and-forget 入 System2 队列，由 `client.digest()` 或调度触发加工。

---

## 五种用法

### 1. SDK

见上「快速开始」。`MemoryClient` 提供 `add / search / get / list / update / delete / delete_bulk / digest`，全异步。

### 2. REST（`dual-mem serve`）

```bash
dual-mem serve --host 0.0.0.0 --port 8000
```

契约对齐 hy-api：`POST /v1/memories/`、`POST /v1/memories/search`、`GET /v1/memories/`、`GET|DELETE /v1/memories/{id}`、`DELETE /v1/memories/`（需 `confirm=true`），外加 `/health` `/ping` `/info`。开启鉴权时需 `Authorization: Bearer <appkey>` 且 `app_id` 在白名单内。

```bash
curl -X POST http://localhost:8000/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{"content": "用户喜欢喝咖啡", "app_id": "default", "user_id": "u1"}'
```

### 3. MCP（`dual-mem mcp`）

```bash
dual-mem mcp   # 通过 stdio 暴露 MCP server
```

暴露工具：`memory_add / memory_search / memory_get / memory_list / memory_delete`，可直接接入支持 MCP 的 Agent 客户端。`memory_search` 返回结果按 profile/proactive/normal 三路分组，演化过的记忆带 `evolution_chain`。

### 4. CLI

```bash
dual-mem add --content "用户喜欢喝咖啡" --app-id default --user-id u1
dual-mem search "用户的饮品偏好" --app-id default --user-id u1
dual-mem list --app-id default --user-id u1
dual-mem get <memory_id>
dual-mem delete <memory_id>
dual-mem digest          # 触发 System2 沉淀（ultra）
```

### 5. Skill

`skills/dual-mem/SKILL.md` 指导 Agent 何时、如何通过 CLI 或 MCP 读写记忆（调用范式、三路分组用法、演化链解读），可直接装入支持 Skill 的 Agent。

---

## 架构

### 8 层记忆框架（L0–L7）

| 层 | 名称 | 含义 |
|---|---|---|
| L0 | BASIC_INFO | 结构化画像（name/age/location/occupation/employer），diff-only 演化链 |
| L1 | RAW | 原始对话（pro/ultra 写后转 SHADOW，避免与 L2/L4 双重命中） |
| L2 | FACT | 客观事件、经历、计划 |
| L3 | SUMMARY | 长内容（≥500 字）摘要 |
| L4 | IDENTITY | 偏好、态度、价值观、人格特质 |
| L5 | KNOWLEDGE | 领域知识 |
| L6 | SCHEMA | System2 提炼的行为模式（场景+模式+洞察） |
| L7 | INTENTION | System2 推断的具体未来意图 |

### 写侧 System1（集中 LLM）

```
add → 写 L1_RAW → Extractor(identity/facts JSON + L0 tool)
                → Summarizer(≥500 字才出 L3)
                → Reconciler(search_query 改写 → 双路检索 → 判 ADD/SUPERSEDE/DELETE)
                → L1 转 SHADOW
```

### 异步 System2（仅 ultra）

```
digest() → 取 fresh facts → 两阶段 DBSCAN 聚类 → 单次 LLM 出 ops 数组
        → ToolExecutor 落库 create_schema(L6)/create_intention(L7)/add_evidence/add_edge
        → CrossDomainSweeper：基础 Schema ≥5 → 升维出核心 Schema（CROSS_ABSTRACTS_TO）
```

### 演化链

Reconciler 在写侧显式建 `supersedes`/`superseded_by` 双向指针；`DELETE` 不真删（`is_latest=False` + `SHADOW`），信息无损。读侧命中链中任一节点 → 回溯到链头（is_latest）→ 附 `evolution_chain`（最新→最旧）。**不依赖相似度聚类。**

### 三路召回（零 LLM）

| 路 | 召回层 | 说明 |
|---|---|---|
| profile | L0 / L4 / L6 | 稳定画像；identity 40% / schema 40% / 自由 20% 配额，`profile_limit=-1` 全量 |
| proactive | L7 | 推断意图；`intention_limit=0` 时恒空（默认关） |
| normal | L2 / L5 / L3 / L1 | 普通事实知识；受 `limit` 与 `min_score` 约束，BM25+RRF 重排 |

> 返回的 profile/proactive/normal 是**路由分组**，不是层；真实层看每条记忆的 `category` 字段。

---

## 与 Hy-Memory 的差异

本项目复现的是 Hy-Memory 的**能力与行为**，不是源码拷贝。主要差异：

1. **8 层命名对齐代码而非官网**：官网文档 L1–L6 命名相对源码有 ±1 位移，本项目弃用官网命名，采用源码语义的 `L0_BASIC_INFO`…`L7_INTENTION`。
2. **REST 契约对齐 hy-api 文档**：源码 REST 是无鉴权的 stdlib `HTTPServer` + 自定义 `/api/v1/add` 内部协议；本项目用 FastAPI 重写为文档约定的 `/v1/memories/` 契约 + Bearer/appkey 鉴权 + 统一错误码。
3. **新增 MCP / CLI / Skill / 鉴权 / 测试**：源码这五块完全没有，本项目全新构建（含 128 单测 + e2e smoke）。
4. **时间查询解析增强**：源码默认 legacy reader 不解析时间词（横评点名短板，「昨天聊了啥」类几乎全错）；本项目默认给 reader 接入中文相对时间解析（`昨天/上周/X月X日` → `created_after` 区间）。
5. **System2 简化为单轮 ops**：源码 System2 是 8 工具 ReAct 多轮循环；本项目按计划简化为「单次 LLM 输出 ops JSON 数组 → 程序串行执行」（有意偏离，换取可控与可测）。
6. **配置收敛**：源码几十个分散的 `MEMORY_*` 环境变量收敛为 `DUAL_MEM_` 前缀的 pydantic-settings，`mode` 自动派生其余开关。
7. **砍掉死代码**：源码 4 套 reader（legacy/hybrid/hybrid_tag/exhaustive）、`rdb.py` 桩、`scorer(V1)`、`emotion_analyzer`、`intention_detector` 等只留一套增强 reader。

---

## 已知短板 / 限制

- **supersedes 不跨层纠错**：演化链是**层内**链路（L4↔L4、L2↔L2）。L0 若被误抽（如一句话误判用户名），不会被 L2/L4 纠正 —— 如实保留源码行为。
- **System2 单轮 ops 表达力弱于 ReAct**：无法多轮取证/反思，复杂认知结构的提炼不如源码多轮循环细腻。
- **tag_index 图桥接为简化版**：跨域升维（CrossDomainSweeper）用「基础 Schema 数 ≥5 触发 + 一次 LLM 归纳」，而非源码的行为升维 embed + 矩阵碰撞 + Union-Find。
- **L0 抽取依赖 LLM 质量**：结构化画像的准确性取决于 extract 阶段 LLM 的判断，弱模型可能误抽/漏抽。
- **Kuzu 无原生向量索引**：L6/L7 的向量召回在 Python 侧做 cosine 全量比对，数据量大时性能受限。
- **摘要触发阈值**：L3 仅在 content ≥ 500 字才生成；单条短文本 add 几乎不产摘要（messages 形态更易触发）。

---

## 测试与开发

```bash
# 全量单测（已全程 mock LLM/Embed，不真连网络）
python -m pytest tests/ -q

# 静态检查（__init__ 参数 / 方法名 / 模块路径 / 导出列表一致性）
python ~/.agents/rules/check_ast.py .

# 端到端 smoke（三档 + REST + MCP + CLI，无需 API key）
python tmp/m7_e2e_demo.py
```

测试遵循 TDD，`tests/conftest.py` 注入确定性 `FakeEmbed` + 脚本化 `FakeLLMClient`，绝不真连 LLM/Embed。
