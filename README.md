<h1 align="center">dual-mem</h1>

<p align="center"><b>当场记下事实，事后看出规律。</b><br />System 1 在对话里写下事实和变化，dual（System 2）再把它们归纳成稳定规律。MemHub 让多个 Agent 共用这一份记忆。</p>

<p align="center">
  <a href="https://badge.fury.io/py/dual-mem"><img src="https://badge.fury.io/py/dual-mem.svg" alt="PyPI version" /></a>
  <a href="https://github.com/shibing624/dual-mem/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License Apache 2.0" /></a>
  <a href="https://github.com/shibing624/dual-mem/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-green.svg" alt="Python 3.11+" /></a>
  <a href="https://github.com/shibing624/dual-mem/issues"><img src="https://img.shields.io/github/issues/shibing624/dual-mem.svg" alt="GitHub issues" /></a>
  <a href="#社区与支持"><img src="https://img.shields.io/badge/wechat-group-green.svg?logo=wechat" alt="Wechat Group" /></a>
</p>

<p align="center">简体中文 | <a href="https://github.com/shibing624/dual-mem/blob/main/README_EN.md">English</a> | <a href="https://shibing624.github.io/dual-mem">在线文档</a></p>

## 为什么选 dual-mem

你已经告诉 AI 三次「我现在不吃辣了」，它还在推荐川菜馆。聊了几十轮，它背得出每一句原话，却说不出你做事的稳定习惯。在 Cursor 里交代过的代码规范，Codex 并不知道。

### 1. 🏆 同一个模型，记得更准、写得更快

抽取、作答、评分用同一个模型（Qwen3.8-27B），同一套 embedding（bge-m3），同一批题目——**只换记忆系统**：

| 记忆系统 | LoCoMo<br>(308 题) | LongMemEval<br>(24 题) | PersonaMem-v2<br>(476 题) | 单篇长对话写入耗时<br>(秒，越低越快) |
|---|:---:|:---:|:---:|:---:|
| **dual-mem** | **74.7** | **91.7** | **50.0** | **503** |
| hy-memory 2.0.4 | 66.6 | 70.8 | 43.5 | 2485 |
| mem0 | 65.6 | 83.3 | 47.9 | 808 |

**三场公开评测都排第一，写入大约是 hy-memory 的五分之一。** 协议、显著性、题型细分和复现命令见 [评测说明](https://github.com/shibing624/dual-mem/blob/main/docs/benchmarks.md)。

### 2. 🧠 两套系统，这就是 dual

名字里的 dual 指 chat 里的两套记忆，也是论文里要写的方法层。

- **System 1（默认 `system1`）**：对话进行时写入。原话、事实、个人画像分层存放；你改了主意、换了城市或更新了方案，它追加新版本并保留历史，回答用最新的。
- **dual / System 2（`mode="dual"`，再调用 `digest()`）**：会话告一段落时，跨多条证据归纳行为规律和长期画像（L3）。

对应示例：[`examples/02_dual.py`](https://github.com/shibing624/dual-mem/blob/main/examples/02_dual.py)。

### 3. 🔗 多个 Agent，一份记忆

MemHub 把上面这两套系统接到本机上的多个工具。Claude Code、Codex、Cursor、Hermes、Agentica 读写同一份记忆，数据默认留在你的电脑上。

```bash
pip install "dual-mem[all]"

dual-mem hub install-service       # 后台常驻，开机自启
dual-mem hub install-host all      # 一键接入本机已安装的 AI 工具
dual-mem hub scan --source all     # 把过去的对话也变成记忆

dual-mem hub recall "我之前做过哪些技术选择"
```

零配置发现本机服务，对话先存下来立刻返回，整理在后台进行。也可以在你自己的服务器上只跑一个 Core，多台电脑连同一份记忆，见 [服务器搭建](https://github.com/shibing624/dual-mem/blob/main/docs/memhub-server.md)。本机用法见 [MemHub 文档](https://github.com/shibing624/dual-mem/blob/main/docs/memhub.md)。

### 4. 🛠️ 工程排障也能留下来

Coding 记忆吃的是 coding agent 的工具轨迹，不是一份源代码。`add(messages=...)` 里要有 `role="tool"` 或 assistant 的 `tool_calls`，JEV 再把整段判成工程场景，才会抽出根因、方案和文件。`search_coding` 只查这条库。用法见 [快速开始 · Coding 记忆](https://github.com/shibing624/dual-mem/blob/main/docs/getting-started/quickstart.md#coding)。

## 安装

```bash
pip install dual-mem
```

含 REST / CLI / MCP / MemHub：`pip install "dual-mem[all]"`。首次运行会把包内 [`config.default.yaml`](https://github.com/shibing624/dual-mem/blob/main/dual_mem/config.default.yaml) 复制成 `~/.dual_mem/config.yaml`。启动要三把钥匙，缺一就拒绝构造：

```yaml
jev_api_key: ""    # 写入裁决。System One，不是 Chat Completions
jev_endpoint: https://api.typesafe.ai/v1/systemone

llm_api_key: ""    # 抽取与 System 2。OpenAI 兼容，地址可改
llm_base_url: https://api.openai.com/v1

embed_api_key: ""  # 向量索引与检索。OpenAI 兼容，地址可改
embed_base_url: https://api.openai.com/v1
```

三把钥匙和三个地址都可以改成环境变量：`DUAL_MEM_JEV_API_KEY` / `DUAL_MEM_JEV_ENDPOINT`、`DUAL_MEM_LLM_API_KEY` / `DUAL_MEM_LLM_BASE_URL`、`DUAL_MEM_EMBED_API_KEY` / `DUAL_MEM_EMBED_BASE_URL`。

## 快速开始

### 日常记忆

```python
from dual_mem import SyncMemoryClient

with SyncMemoryClient() as memory:
    memory.add(content="我住在上海，最近开始不吃辣了。", user_id="alice")
    memory.add(content="下个月我要搬去北京工作。", user_id="alice")

    res = memory.search(query="给 alice 推荐一家餐厅", user_id="alice")
    for m in res.memories.profile + res.memories.normal:
        print(m.content)
```

它会记住 alice 不吃辣、即将搬去北京，并且知道「上海」已经是过去时。

### Coding 记忆

`coding_enabled=True` 之后，只有 `messages` 里出现工具轨迹才会进 Coding 库：某条是 `role="tool"`，或者 assistant 带了 `tool_calls`。这是结构条件。接着 JEV 判断整段是不是工程场景，是才抽出 `task` / `solution` / `files`。

把几个 `.py` 文件的正文放进 `add(content=...)` 或普通 user 消息，没有工具轨迹，走的是 chat 记忆。`search()` 能搜到当事实记下的内容，`search_coding` 不会返回。

```python
from dual_mem import SyncMemoryClient
from dual_mem.config import Settings

with SyncMemoryClient(settings=Settings(coding_enabled=True)) as memory:
    memory.add(
        messages=[
            {"role": "user", "content": "高并发写入 SQLite 偶发 database is locked，帮我排查。"},
            {
                "role": "assistant",
                "content": "我先读取存储层代码查看事务开启方式。",
                "tool_calls": [{"id": "t1", "name": "read_file", "arguments": {"path": "app/db.py"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": "conn = sqlite3.connect(db_path, timeout=5.0)  # default DEFERRED transaction",
            },
            {
                "role": "assistant",
                "content": "原因：DEFERRED 事务先读后写，升级写锁时若另一连接已持有写锁，SQLite 直接返回 SQLITE_BUSY 而不等待 timeout。改为 BEGIN IMMEDIATE 提前获取写锁，并将 timeout 调至 30s。",
            },
        ],
        user_id="alice",
    )

    hits = memory.search_coding(query="SQLite 并发写入报 database is locked 怎么解决", user_id="alice")
    for h in hits:
        print(h["task"], "->", h["solution"], h["files"])
```

### 示例

两条最值得看的：[`examples/02_dual.py`](https://github.com/shibing624/dual-mem/blob/main/examples/02_dual.py) 是 `digest()` 归纳 L3，[`examples/06_coding_memory.py`](https://github.com/shibing624/dual-mem/blob/main/examples/06_coding_memory.py) 是工具轨迹写入和 `search_coding`。REST、CLI、生命周期在 [`examples/`](https://github.com/shibing624/dual-mem/tree/main/examples) 目录里。异步客户端见 [快速开始](https://github.com/shibing624/dual-mem/blob/main/docs/getting-started/quickstart.md)。

## 功能特性

**记忆**

- **版本演化** — 「住上海 → 搬北京」是一次更新，旧版本留作历史，回答默认用最新的
- **分层存放** — 原话（L0）、事实与意图（L1）、画像（L2）、跨证据规律（L3）分开，按需取用
- **决策模型(Jev)把关写入** — 新信息、重复，还是更新，由轻量模型判断
- **原话可查** — 整理过的记忆之外，原始对话始终保留

**接入**

- **Python SDK** — `pip install dual-mem`
- **MemHub** — `dual-mem hub install-host all`
- **MCP / REST / CLI** — `dual-mem-mcp`、`dual-mem serve`、`dual-mem add` / `dual-mem search`
- **框架插件** — Agentica、Claude Code Hook、Hermes、OpenClaw，见 [集成文档](https://github.com/shibing624/dual-mem/blob/main/docs/integrations.md)

## Roadmap

- [x] System 1 分层记忆 + 版本演化
- [x] dual / System 2：显式 `digest()` 归纳 L3
- [x] 专用决策模型把关写入
- [x] MemHub：5 个 AI 工具共享记忆、历史会话导入、后台整理
- [x] 云主机单实例：多台电脑连同一份记忆
- [x] LoCoMo / LongMemEval / PersonaMem-v2 同场评测领先
- [ ] MemHub 接入更多工具：OpenCode、WorkBuddy、Gemini CLI、Windsurf、Trae 等，目标 10+ 种
- [ ] 云端 HTTP MCP：远程 Agent 直接连接托管记忆
- [ ] 多跳推理类问题的检索增强（LoCoMo multi-hop 仍有提升空间）
- [ ] 多设备记忆同步：离线可用、联网自动合并
- [ ] 记忆可视化管理：查看、编辑、删除你的记忆
- [ ] 隐私分级：敏感记忆只留本机
- [ ] 补充更多公开评测（CodeMemoryBench 等，待数据公开）

欢迎在 [Issues](https://github.com/shibing624/dual-mem/issues) 投票或提需求。

## News

- [2026/10/23] **v0.1.5**：修复 Coding 记忆接真实模型时写不进去的问题。抽取改为返回 `{"candidates": [...]}` 对象，适配默认 `llm_json_mode`；去掉 coverage 硬门槛；同一批写入可以覆盖同一条旧方案；裁决只看当前版本边界内的记录；同一用户的并发写入按 scope 串行。快速上手补充 Coding 记忆章节，`examples/06_coding_memory.py` 增加方案更新后只返回当前版本的示例。详见 [Release-v0.1.5](https://github.com/shibing624/dual-mem/releases/tag/v0.1.5)。
- [2026/10/23] **v0.1.4**：新增 MemHub 本地单 Core 跨工具记忆中枢（支持 Claude Code / Codex / Cursor / Hermes / Agentica 会话增量导入、后台持久队列写入、开机自启与云端隧道连接）；升级 System 2 v3 聚类认知与版本桥召回；补充 Coding 记忆示例并精简无收益配置项；在 LoCoMo（74.7）、LongMemEval（91.7）、PersonaMem-v2（50.0）同场评测中全面领先 hy-memory 2.0.4 与 mem0。详见 [Release-v0.1.4](https://github.com/shibing624/dual-mem/releases/tag/v0.1.4)。
- [2026/10/15] **v0.1.3**：升级为 L0_RAW / L1_FACT / L2_PROFILE / L3_DERIVED 四层记忆架构与 JEV 专用决策模型写路径，支持 target-first 关系裁决、版本演化链与历史回溯；新增 Agentica / Claude Code / Hermes / OpenClaw 原生生态集成与 Coding 记忆子系统（存储格式不兼容旧版，需使用新目录）。详见 [Release-v0.1.3](https://github.com/shibing624/dual-mem/releases/tag/v0.1.3)。

<details>
<summary>更早版本</summary>

- [2026/06/20] **v0.1.2**：新增 `SyncMemoryClient` 同步客户端封装，支持首次运行自动生成 `~/.dual_mem/config.yaml` 默认配置，完善 REST / MCP / CLI 统一契约与在线文档站点。详见 [Release-v0.1.2](https://github.com/shibing624/dual-mem/releases/tag/v0.1.2)。
- [2026/06/19] **v0.1.1**：发布双系统分层记忆 SDK，支持语义 + BM25 + RRF 混合检索及 CLI / REST / MCP 接口。详见 [Release-v0.1.1](https://github.com/shibing624/dual-mem/releases/tag/v0.1.1)。
- [2026/06/18] **v0.1.0**：首版开源。

</details>

完整变更见 [CHANGELOG](https://github.com/shibing624/dual-mem/blob/main/CHANGELOG.md)。

## 文档

完整使用文档：**[https://shibing624.github.io/dual-mem](https://shibing624.github.io/dual-mem)**

| 文档 | 内容 |
|---|---|
| [安装与配置](https://github.com/shibing624/dual-mem/blob/main/docs/getting-started/installation.md) | extras、配置文件、凭证与优先级 |
| [快速开始](https://github.com/shibing624/dual-mem/blob/main/docs/getting-started/quickstart.md) | 同步 / 异步客户端、多轮写入、Agent 集成 |
| [架构说明](https://github.com/shibing624/dual-mem/blob/main/docs/architecture.md) | SDK 与各接入层的关系、存储 |
| [四层记忆](https://github.com/shibing624/dual-mem/blob/main/docs/memory_layers.md) | L0 原文 / L1 事实 / L2 画像 / L3 推导 |
| [写入决策设计](https://github.com/shibing624/dual-mem/blob/main/docs/jev-memory-design.md) | 决策模型动作与关系、版本链、并发 |
| [MemHub](https://github.com/shibing624/dual-mem/blob/main/docs/memhub.md) | Core 服务、宿主接入、云端单实例 |
| [MCP / REST](https://github.com/shibing624/dual-mem/blob/main/docs/mcp_integration.md) | 工具 ↔ HTTP ↔ SDK 对照 |
| [生态集成](https://github.com/shibing624/dual-mem/blob/main/docs/integrations.md) | Agentica / Claude Code / Hermes / OpenClaw |
| [评测说明](https://github.com/shibing624/dual-mem/blob/main/docs/benchmarks.md) | 协议、题集、复现与结果细分 |
| [示例说明](https://github.com/shibing624/dual-mem/blob/main/examples/README.md) | 6 个可运行的端到端示例脚本说明 |

本地预览：`pip install -r requirements-docs.txt && mkdocs serve`

## 社区与支持

> 如果 dual-mem 帮到了你，欢迎点个 ⭐ Star，让更多人看到。

- **GitHub Issues** — [提交 issue](https://github.com/shibing624/dual-mem/issues)
- **微信群** — 添加微信号 `xuming624`，备注 "llm"，加入技术交流群

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" alt="微信群二维码" />

## 引用

如果您在研究中使用了 dual-mem，请引用：

> Xu, M. (2026). dual-mem: Dual-System Layered Memory SDK for LLM Applications. GitHub. https://github.com/shibing624/dual-mem

```bibtex
@misc{xu2026dualmem,
  author    = {Xu, Ming},
  title     = {dual-mem: Dual-System Layered Memory SDK for LLM Applications},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/shibing624/dual-mem}
}
```

## 许可证

[Apache License 2.0](https://github.com/shibing624/dual-mem/blob/main/LICENSE)

## 贡献

欢迎贡献，见 [CONTRIBUTING.md](https://github.com/shibing624/dual-mem/blob/main/CONTRIBUTING.md)。

## 致谢

- [Chroma](https://github.com/chroma-core/chroma)
- [Kuzu](https://github.com/kuzudb/kuzu)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [mem0](https://github.com/mem0ai/mem0)
