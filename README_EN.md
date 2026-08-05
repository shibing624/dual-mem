[**🇨🇳中文**](https://github.com/shibing624/dual-mem/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/dual-mem/blob/main/README_EN.md)

# dual-mem: Dual-System Layered Memory SDK
[![PyPI version](https://badge.fury.io/py/dual-mem.svg)](https://badge.fury.io/py/dual-mem)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.11%2B-green.svg)](pyproject.toml)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/dual-mem.svg)](https://github.com/shibing624/dual-mem/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#community--support)

**dual-mem** helps agents **actually remember users** — not by dumping chat logs, but like people do: **capture today’s conversation quickly, then consolidate it into profile, habits, and plans**. Layered storage, preference updates with history, and retrieval that surfaces what matters. Plug in via **SDK / REST / MCP / CLI / Skill**.

## Memory architecture (L0–L7)

dual-mem implements **seven memory layers (L0–L4, L6, L7)** — inspired by how the brain moves from raw experience → encoding → consolidation → abstraction: from chat traces to profile, habits, and plans, with **evolution chains** (preference updates keep history instead of blind overwrite):

| Layer | What it holds | Example |
|---|---|---|
| **L0 Profile** | Structured basics | Name, age, city |
| **L1 Raw traces** | Full interaction history | Multi-turn chat (archived after extraction) |
| **L2 Facts** | Verifiable statements | “Flying to Beijing next week” |
| **L3 Summaries** | Long-text compression | Key points from a long note |
| **L4 Identity & prefs** | Stable “who you are” | “Main language is Python” |
| **L6 Patterns** | Habits from many facts | “Lists everything before trips; loves organizing” |
| **L7 Intentions** | Plans and goals | “Training for a marathon” |

When preferences change (Java → Python), old entries aren’t erased — an **evolution chain** keeps history; reads return the **latest version plus the trail**.

## Dual system: short-term × deep memory

| | **System1 · short-term** | **System2 · deep memory** |
|---|---|---|
| **Like** | Hear and note it now | Sleep on it; understand deeper over time |
| **When** | Every `add`, ready in seconds | Explicit `digest()` in `dual` mode |
| **Produces** | L0–L4: facts, prefs, profile | L6 patterns, L7 intentions, linked knowledge |
| **How to enable** | `mode="system1"` (default) | `mode="dual"` |

```
add conversation ──▶ System1 short-term memory (usable immediately)
                         │
                         └──▶ System2 deep memory (dual: consolidate on digest)
```

Recall groups results into **profile / proactive / normal**; hybrid semantic + keyword search; **no extra LLM on read** (embedding only).

## Why dual-mem

| | |
|---|---|
| **Easy start** | `SyncMemoryClient`: write + search in a few lines; config auto-created on first run |
| **Agent-ready** | Batch multi-turn `messages`; Agentica integration in [Quickstart](docs/getting-started/quickstart.md) |
| **Evolving memory** | Preference updates via chains, not blind overwrite |
| **Five entry points** | One `MemoryClient`; optional REST / MCP / CLI extras |

## 🔥 News

- [2026/06/20] **v0.1.2**: MemoryOperations for REST/MCP; SyncMemoryClient; CLI messages; post-extract embed batching; MCP bootstrap fixes and docs.
- [2026/06/19] **v0.1.1**: Dependencies split into `[api]` / `[cli]` / `[mcp]` extras; hybrid reader V2 became the default.
- [2026/06/18] **v0.1.0**: Initial release — system1 / dual modes, evolution chains, REST `/v1/memories/` contract, MCP tools.

## Architecture at a glance

```
  Your agent / app
        │  SDK · REST · MCP · CLI · Skill
        ▼
   MemoryClient (System1 write / search / evolution chains)
        │
   Vector + graph + local storage
```

Implementation details (Extractor commit decisions, reconcile, hybrid retrieval, etc.): [architecture.md](docs/architecture.md) · [docs site](https://shibing624.github.io/dual-mem/architecture/).

## Installation

Requires Python ≥ 3.11.

```bash
pip install dual-mem              # core SDK
pip install dual-mem[all]         # + REST / CLI / MCP
```

Development install:

```bash
git clone https://github.com/shibing624/dual-mem.git
cd dual-mem
pip install -e ".[dev]"
```

| Extra | Dependencies | Use case |
|---|---|---|
| `api` | fastapi, uvicorn | `dual-mem serve`, REST API |
| `cli` | typer | `dual-mem add/search/...` |
| `mcp` | mcp | `dual-mem-mcp`, Cursor MCP |
| `all` | all of the above | full stack |

## Quick Start

Scripts / sync apps (recommended):

```python
from dual_mem import SyncMemoryClient

with SyncMemoryClient(mode="system1", storage_dir="./.dual_mem_data") as client:
    client.add(
        content="My favorite language is Java. I've used it for 5 years.",
        user_id="alice",
    )
    res = client.search(
        query="programming language preference",
        user_id="alice",
    )
    for m in res.memories.profile:
        print(m.content)
```

FastAPI / asyncio agents:

```python
import asyncio
from dual_mem import MemoryClient


async def main():
    client = MemoryClient(mode="system1", storage_dir="./.dual_mem_data")
    await client.add(content="...", user_id="alice")
    await client.aclose()


asyncio.run(main())
```

Config lives at `~/.dual_mem/config.yaml` (**auto-created on first startup**). **Both LLM and embedding API keys are required.**

→ Parameters, multi-turn `messages`, `aclose()`, and agent integration patterns: **[Quickstart](docs/getting-started/quickstart.md)** ([online](https://shibing624.github.io/dual-mem/getting-started/quickstart/)).

## Features

- **Seven implemented layers + evolution chains** — from raw chat to patterns and intentions; preference history preserved
- **System1 short-term memory** — structured on write; the Extractor decides whether to commit derived memories
- **System2 deep memory** (dual) — explicit `digest()` consolidates L6/L7 at a business boundary
- **Hybrid retrieval** — semantic + keyword; profile / proactive / normal routes
- **Sync & async SDK** — `SyncMemoryClient` for scripts; `MemoryClient` for FastAPI / agents
- **OpenAI-compatible** — any LLM / embedding endpoint
- **Single-product default** — omit `app_id` to use `default_app_id` (`"default"`); `user_id` is the primary isolation key

## Modes

| | **system1** (default) | **dual** |
|---|---|---|
| Short-term memory | ✓ | ✓ |
| Deep memory | — | ✓ (explicit `digest()` consolidation) |
| Best for | Assistants, support bots, quick integration | Deeper user modeling, intentions, behavior patterns |
| Graph links | off | on |

## REST / MCP / CLI

```bash
pip install dual-mem[all]

dual-mem serve --host 0.0.0.0 --port 8000          # REST
dual-mem-mcp                                          # MCP stdio
dual-mem-mcp --transport streamable-http --port 8765 # MCP HTTP /mcp
dual-mem add --content "User likes coffee" --app-id default --user-id u1
dual-mem search "drink preference" --app-id default --user-id u1
dual-mem digest   # dual: trigger System2
```

REST contract: `POST /v1/memories/`, `POST /v1/memories/search`, `GET|DELETE /v1/memories/{id}`, etc. See [MCP integration](docs/mcp_integration.md).

## Examples

See [examples/](https://github.com/shibing624/dual-mem/tree/main/examples) for runnable demos:

| Category | Content |
|----------|---------|
| **SDK basics** | Short-term writes, evolution chains, multi-turn `messages` |
| **Deep memory** | dual mode with explicit consolidation |
| **REST** | FastAPI TestClient against `/v1/memories/` |
| **CLI** | `dual-mem` add / search / list / delete |

[→ Full examples README](https://github.com/shibing624/dual-mem/blob/main/examples/README.md)

```bash
python examples/01_system1.py
python examples/02_dual.py
python examples/03_rest_api.py
python examples/04_cli.py
```

> Examples call real LLM + embedding APIs (small cost).

## Documentation

Full docs: **https://shibing624.github.io/dual-mem**

Local preview:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

## Community & Support

- **GitHub Issues** — [open an issue](https://github.com/shibing624/dual-mem/issues)
- **WeChat group** — add WeChat ID `xuming624`, note "llm" to join the tech community

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" />

## Citation

If you use dual-mem in research, please cite:

> Xu, M. (2026). dual-mem: Dual-System Layered Memory SDK for LLM Applications. GitHub. https://github.com/shibing624/dual-mem

## License

[Apache License 2.0](LICENSE)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- [chromadb/chroma](https://github.com/chroma-core/chroma) — vector store
- [kuzudb/kuzu](https://github.com/kuzudb/kuzu) — graph store
- [tiangolo/fastapi](https://github.com/tiangolo/fastapi) — REST layer
- [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) — MCP integration
- [mem0ai/mem0](https://github.com/mem0ai/mem0) — memory SDK ecosystem
