[**🇨🇳中文**](https://github.com/shibing624/dual-mem/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/dual-mem/blob/main/README_EN.md) | [**📖Documentation**](https://shibing624.github.io/dual-mem)

<div align="center">

# dual-mem

**Memory That Evolves With You — Across Every AI Tool.**

AI that actually remembers you — it updates automatically when your preferences change, and still knows you when you switch tools.

[![PyPI version](https://badge.fury.io/py/dual-mem.svg)](https://badge.fury.io/py/dual-mem)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/shibing624/dual-mem/blob/main/LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://github.com/shibing624/dual-mem/blob/main/pyproject.toml)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/dual-mem.svg)](https://github.com/shibing624/dual-mem/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#community--support)

</div>

---

You've told your assistant three times that you no longer eat spicy food, and it still recommends a Sichuan restaurant.
The coding conventions you explained in Cursor are unknown to Codex; the stack you settled on in Claude Code has to be re-explained in Hermes.

**dual-mem solves these problems for good:**

- 🧠 **Remembers accurately** — instead of piling up chat logs, it organizes raw dialogue, explicit facts, and user profile in separate layers; when you change your mind, move cities, or switch stacks, it appends a new version and keeps history rather than contradicting itself.
- 🔗 **Remembers everywhere** — MemHub lets Claude Code, Codex, Cursor, Hermes, and Agentica share one local personal memory. Your memory is a long-term asset that no single tool locks in.
- 🛠️ **Remembers engineering work** — beyond daily preferences, it extracts reusable debugging lessons and code solutions from tool-call sessions so you never debug the same issue twice.

## Results: head-to-head, same setup

The same model (Qwen3.8-27B) does extraction, answering and grading; same embedding model (bge-m3), same questions — **only the memory system changes**:

| Memory system | LoCoMo<br>(308 q) | LongMemEval<br>(24 q) | PersonaMem-v2<br>(476 q) | Ingestion time per long dialogue<br>(s, lower is better) |
|---|:---:|:---:|:---:|:---:|
| **dual-mem** | **74.7** | **91.7** | **50.0** | **503** |
| hy-memory 2.0.4 | 66.6 | 70.8 | 43.5 | 2485 |
| mem0 | 65.6 | 83.3 | 47.9 | 808 |

> Ingestion time is the median time to ingest one LoCoMo multi-session long conversation from scratch. Protocol, significance tests, per-category results and reproduction commands: [Benchmarks](https://github.com/shibing624/dual-mem/blob/main/docs/benchmarks.md).

## Get started in 3 minutes

```bash
pip install dual-mem
```

The first run creates `~/.dual_mem/config.yaml`; fill in API keys for your model services (any OpenAI-compatible endpoint).

### 1. Daily memory & preference evolution

```python
from dual_mem import SyncMemoryClient

with SyncMemoryClient() as memory:
    memory.add(content="I live in Shanghai and recently stopped eating spicy food.", user_id="alice")
    memory.add(content="I'm moving to Beijing for work next month.", user_id="alice")

    res = memory.search(query="recommend a restaurant for alice", user_id="alice")
    for m in res.memories.profile + res.memories.normal:
        print(m.content)
```

It remembers that alice no longer eats spicy food and is about to move to Beijing — and it knows "Shanghai" is becoming the past.

### 2. Coding memory: reusable debugging & engineering lessons

With `coding_enabled=True`, when `messages` include tool calls (assistant `tool_calls` or `role="tool"` results), dual-mem automatically detects engineering sessions and extracts structured `task` / `solution` / `files` memories. A changed solution is appended as a new version that retires the old one, so `search_coding` returns only what currently applies (details: [Quickstart · Coding memory](https://github.com/shibing624/dual-mem/blob/main/docs/getting-started/quickstart.md#coding)):

```python
from dual_mem import SyncMemoryClient
from dual_mem.config import Settings

with SyncMemoryClient(settings=Settings(coding_enabled=True)) as memory:
    memory.add(
        messages=[
            {"role": "user", "content": "Fix intermittent sqlite3.OperationalError: database is locked under concurrent writes."},
            {
                "role": "assistant",
                "content": "Let me inspect how transactions are opened in the vector store.",
                "tool_calls": [{"id": "t1", "name": "read_file", "arguments": {"path": "app/db.py"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "t1",
                "content": "conn = sqlite3.connect(db_path, timeout=5.0)  # default DEFERRED transaction",
            },
            {
                "role": "assistant",
                "content": "A DEFERRED transaction that reads first gets SQLITE_BUSY immediately (no timeout wait) when upgrading to a write lock another connection holds. Use BEGIN IMMEDIATE and increase timeout to 30s.",
            },
        ],
        user_id="alice",
    )

    hits = memory.search_coding(query="how to fix SQLite database is locked on concurrent writes", user_id="alice")
    for h in hits:
        print(h["task"], "->", h["solution"], h["files"])
```

### 3. Runnable examples (`examples/`)

The [`examples/`](https://github.com/shibing624/dual-mem/tree/main/examples) directory contains end-to-end runnable scripts covering every major workflow:

| Script | What it covers | Command |
|---|---|---|
| [`examples/01_system1.py`](https://github.com/shibing624/dual-mem/blob/main/examples/01_system1.py) | Single-turn & multi-turn writes, version evolution chains, semantic search, history listing | `python examples/01_system1.py` |
| [`examples/02_dual.py`](https://github.com/shibing624/dual-mem/blob/main/examples/02_dual.py) | `dual` mode writes + `digest()` for cross-evidence L3 schemas and proactive intentions | `python examples/02_dual.py` |
| [`examples/03_rest_api.py`](https://github.com/shibing624/dual-mem/blob/main/examples/03_rest_api.py) | FastAPI `/v1/memories/...` CRUD, versioned updates and archiving | `python examples/03_rest_api.py` |
| [`examples/04_cli.py`](https://github.com/shibing624/dual-mem/blob/main/examples/04_cli.py) | `dual-mem` CLI for adding, searching and querying raw conversations | `python examples/04_cli.py` |
| [`examples/05_lifecycle.py`](https://github.com/shibing624/dual-mem/blob/main/examples/05_lifecycle.py) | `add_raw`, `distill`, `search_conversation`, `update` and `delete` lifecycle | `python examples/05_lifecycle.py` |
| [`examples/06_coding_memory.py`](https://github.com/shibing624/dual-mem/blob/main/examples/06_coding_memory.py) | Coding memory with `coding_enabled=True`: tool-call dialogue ingestion and `search_coding` | `python examples/06_coding_memory.py` |

Building your own agent? See the [Quickstart](https://github.com/shibing624/dual-mem/blob/main/docs/getting-started/quickstart.md) and [Examples Guide](https://github.com/shibing624/dual-mem/blob/main/examples/README.md).

## MemHub: shared memory across your AI tools

```text
   Claude Code   Codex   Cursor   Hermes   Agentica
        \          |        |        |       /
         ─────────────  MemHub  ─────────────
                         │
           your personal memory (this machine / your server)
```

```bash
pip install "dual-mem[all]"

dual-mem hub install-service       # runs in the background, starts at login
dual-mem hub install-host all      # connect every supported AI tool on this machine
dual-mem hub scan --source all     # turn your past conversations into memory

dual-mem hub recall "what technical choices have I made before"
```

- **Zero configuration**: tools discover the local memory service automatically — no URLs or keys to paste.
- **Never in your way**: conversations are saved and acknowledged instantly; organizing happens in the background.
- **Your history counts**: import existing sessions in one command, so it knows you from day one.
- **Your data stays yours**: by default it lives only on your computer; you can also host it on your own server and connect several computers to the same memory.

Supported today: **Claude Code, Codex, Cursor, Hermes, Agentica**. More tools are on the [Roadmap](#roadmap). See the [MemHub docs](https://github.com/shibing624/dual-mem/blob/main/docs/memhub.md).

## Who it's for

| You are… | dual-mem helps you… |
|---|---|
| **A heavy AI-tool user** | share your preferences, conventions and project context across tools with MemHub — stop repeating yourself |
| **An agent / app developer** | add long-term memory to your product in three lines; integrate via SDK, REST, MCP or CLI |
| **Building assistants or support bots** | remember each user's profile and how it changes, so answers get better over time |
| **A researcher** | run head-to-head comparisons against mem0 and hy-memory out of the box |

## Why it remembers better

- **Updates instead of confusion**: every memory is versioned. "Lives in Shanghai → moving to Beijing" is an update; the old version is kept as history, and answers use the latest by default.
- **Facts and profile organized separately**: what you said, explicit facts, and who you are live in separate layers and are retrieved as needed.
- **A dedicated decision model guards writes**: a lightweight decision model judges "new, duplicate, or update" — steadier and cheaper than asking a large LLM.
- **The original words are always there**: alongside organized memory, the raw conversation is kept so you can always check the source.

Curious how it works? See the [developer docs](#developer-docs) below.

## Ways to integrate

| Option | Best for | One-liner |
|---|---|---|
| **Python SDK** | your own agent or app | `pip install dual-mem` |
| **MemHub** | daily use across several AI tools | `dual-mem hub install-host all` |
| **MCP** | Cursor, Claude Desktop and other MCP clients | `dual-mem-mcp` |
| **REST API** | other languages or a standalone service | `dual-mem serve` |
| **CLI** | scripts and debugging | `dual-mem add` / `dual-mem search` |
| **Framework plugins** | Agentica, Claude Code hooks, Hermes, OpenClaw | see [Integrations](https://github.com/shibing624/dual-mem/blob/main/docs/integrations.md) |

## Roadmap

- [x] Layered, versioned memory: preference changes are traceable
- [x] Dedicated decision model guarding writes
- [x] MemHub: 5 AI tools share memory, history import, background organizing
- [x] Single-instance server mode: several computers share one memory
- [x] Leads head-to-head on LoCoMo / LongMemEval / PersonaMem-v2
- [ ] MemHub for more tools: OpenCode, WorkBuddy, Gemini CLI, Windsurf, Trae and others — 10+ in total
- [ ] Cloud HTTP MCP: remote agents connect to hosted memory directly
- [ ] Better retrieval for multi-hop questions (LoCoMo multi-hop still has headroom)
- [ ] Multi-device sync: works offline, merges automatically when online
- [ ] Visual memory management: view, edit and delete your memories
- [ ] Privacy tiers: sensitive memories stay on this machine
- [ ] More public benchmarks (CodeMemoryBench and others, once their data is released)

Vote or request features in [Issues](https://github.com/shibing624/dual-mem/issues).

## News

- [2026/10/23] **v0.1.5**: Fixed Coding memory writes against real models. Extraction now returns a `{"candidates": [...]}` object so the default `llm_json_mode` keeps the payload; the coverage hard gate is gone; one batch can supersede the same head; JEV only sees heads inside the draft's version boundary; concurrent writes for one user serialize per scope. The quickstart now has a Coding memory section, and `examples/06_coding_memory.py` shows search returning only the current solution after an update.
- [2026/10/23] **v0.1.4**: Added MemHub local single-Core cross-tool memory hub (incremental session import for Claude Code / Codex / Cursor / Hermes / Agentica, durable background write queue, login service, and SSH/Tailscale tunnel attach); upgraded System 2 v3 clustered cognition and version-bridge recall; added Coding memory runnable demo and removed zero-gain settings; ranked #1 head-to-head on LoCoMo (74.7), LongMemEval (91.7), and PersonaMem-v2 (50.0).
- [2026/10/15] **v0.1.3**: Redesigned around four canonical layers (`L0_RAW` / `L1_FACT` / `L2_PROFILE` / `L3_DERIVED`) and target-first JEV write decisions with linked version history; added native integrations for Agentica / Claude Code / Hermes / OpenClaw and the Coding memory subsystem (incompatible storage schema; use a new `storage_dir`).

<details>
<summary>Earlier releases</summary>

- [2026/06/20] **v0.1.2**: Added `SyncMemoryClient` synchronous wrapper, automatic `~/.dual_mem/config.yaml` bootstrap on first run, unified REST / MCP / CLI contracts, and the online documentation site.
- [2026/06/19] **v0.1.1**: Released dual-system layered memory SDK with hybrid retrieval (semantic + BM25 + RRF) and CLI / REST / MCP interfaces.
- [2026/06/18] **v0.1.0**: First open-source release.

</details>

Full history in the [CHANGELOG](https://github.com/shibing624/dual-mem/blob/main/CHANGELOG.md).

## Developer docs

Online docs: **[https://shibing624.github.io/dual-mem](https://shibing624.github.io/dual-mem)** · Local preview: `pip install -r requirements-docs.txt && mkdocs serve`

| Doc | Contents |
|---|---|
| [Installation & configuration](https://github.com/shibing624/dual-mem/blob/main/docs/getting-started/installation.md) | extras, config file, credentials and precedence |
| [Quickstart](https://github.com/shibing624/dual-mem/blob/main/docs/getting-started/quickstart.md) | sync / async clients, multi-turn writes, agent integration |
| [Architecture](https://github.com/shibing624/dual-mem/blob/main/docs/architecture.md) | how the SDK relates to each frontend; storage |
| [Memory layers](https://github.com/shibing624/dual-mem/blob/main/docs/memory_layers.md) | L0 raw / L1 facts / L2 profile / L3 derived |
| [Write decisions](https://github.com/shibing624/dual-mem/blob/main/docs/jev-memory-design.md) | decision-model actions and relations, version chains, concurrency |
| [MemHub](https://github.com/shibing624/dual-mem/blob/main/docs/memhub.md) | Core service, host integration, server mode |
| [MCP / REST](https://github.com/shibing624/dual-mem/blob/main/docs/mcp_integration.md) | tool ↔ HTTP ↔ SDK mapping |
| [Integrations](https://github.com/shibing624/dual-mem/blob/main/docs/integrations.md) | Agentica / Claude Code / Hermes / OpenClaw |
| [Benchmarks](https://github.com/shibing624/dual-mem/blob/main/docs/benchmarks.md) | protocol, question sets, reproduction, breakdowns |
| [Examples](https://github.com/shibing624/dual-mem/blob/main/examples/README.md) | 6 runnable end-to-end example scripts |

## Community & Support

- **GitHub Issues** — [open an issue](https://github.com/shibing624/dual-mem/issues)
- **WeChat group** — add WeChat ID `xuming624` with the note "llm" to join

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" />

## Citation

If you use dual-mem in your research, please cite:

> Xu, M. (2026). dual-mem: Dual-System Layered Memory SDK for LLM Applications. GitHub. https://github.com/shibing624/dual-mem

## License · Contributing · Acknowledgements

[Apache License 2.0](https://github.com/shibing624/dual-mem/blob/main/LICENSE) · Contributions welcome, see [CONTRIBUTING.md](https://github.com/shibing624/dual-mem/blob/main/CONTRIBUTING.md)

Thanks to [Chroma](https://github.com/chroma-core/chroma), [Kuzu](https://github.com/kuzudb/kuzu), [FastAPI](https://github.com/tiangolo/fastapi), [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) and [mem0](https://github.com/mem0ai/mem0).
