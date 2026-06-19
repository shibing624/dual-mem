[**🇨🇳中文**](https://github.com/shibing624/dual-mem/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/dual-mem/blob/main/README_EN.md)

# dual-mem: Dual-System Layered Memory SDK
[![PyPI version](https://badge.fury.io/py/dual-mem.svg)](https://badge.fury.io/py/dual-mem)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.11%2B-green.svg)](pyproject.toml)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/dual-mem.svg)](https://github.com/shibing624/dual-mem/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#community--support)

**dual-mem** is a **dual-system layered memory SDK** for LLM applications and agents. It turns conversational input into structured, evolvable, retrievable long-term memory, exposed through **SDK / REST / MCP / CLI / Skill**.

| Capability | Description |
|------------|-------------|
| **Evolution chains** | Explicit `supersedes` pointers; read path expands full history (newest→oldest) |
| **System1 write path** | Gate (LLM) → Extract → fast-write L2/L4 → async Reconcile |
| **System2 distillation** | dual mode: DBSCAN + 8-tool ReAct → L6 Schema / L7 Intention |
| **Three-route recall** | profile / proactive / normal; default hybrid reader (zero LLM on read) |
| **Multi-entry** | Five frontends share one `MemoryClient`; optional extras |

## 🔥 News

- [2026/06/19] **v0.1.1**: Attentional Gate is now **LLM-primary** (heuristic fallback); dependencies split into `[api]` / `[cli]` / `[mcp]` extras; hybrid reader V2 by default.
- [2026/06/18] **v0.1.0**: Initial release — system1 / dual modes, evolution chains, REST `/v1/memories/` contract, MCP tools.

## Architecture

dual-mem uses a **core SDK + parallel frontends** layout: `MemoryClient` owns all business logic; CLI / REST / MCP / Skill are thin protocol adapters.

```
        Agent / Cursor / Your app
              │
    ┌─────────┼─────────┬─────────┐
    │  Skill  │   MCP   │  REST   │  CLI     ← frontends (optional extras)
    └─────────┼─────────┴─────────┘
              ▼
       dual_mem.MemoryClient          ← core SDK
              │
    Chroma (vector) + Kuzu (graph) + SQLite (cache/queues)
```

Write / read overview:

```
Write: Gate(1×LLM) → Extract(1×LLM) → fast-write L2/L4 → [async Reconcile / System2]
Read:  QueryUnderstanding → AnchorSearch(5 routes) → GraphExpander → FusionScorer → 3 routes
```

See [docs/architecture.md](docs/architecture.md) and the online [Architecture](https://shibing624.github.io/dual-mem/architecture/) page.

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

```python
import asyncio
from dual_mem import MemoryClient


async def main():
    client = MemoryClient(mode="system1", storage_dir="./.dual_mem_data")

    await client.add(
        content="My favorite language is Java. I've used it for 5 years.",
        app_id="my_app",
        user_id="alice",
    )

    res = await client.search(
        query="programming language preference",
        app_ids=["my_app"],
        user_id="alice",
    )
    for m in res.memories.profile:
        print(m.content)

    await client.aclose()


asyncio.run(main())
```

Copy `config.example.yaml` to `~/.dual_mem/config.yaml`. **Both LLM and embedding API keys are required.**

```yaml
mode: system1
llm_api_key: sk-...
embed_api_key: sk-...
embed_model: text-embedding-3-small
embed_dim: 1536
```

## Features

- **Async-first** — fully async `MemoryClient`; CLI/REST wrap with `asyncio.run`
- **Attentional Gate** — LLM scores novelty / relevance / arousal; heuristic fallback on failure
- **Fast-write + async Reconcile** — low-latency writes; `reconcile_sync=true` for strong consistency
- **Evolution chains** — intra-layer `supersedes` pointers; soft-delete history; `evolution_chain` on read
- **Hybrid reader (default)** — 5-route parallel anchors + 1-hop graph + FusionScorer; Chinese time-word parsing
- **System2 ReAct (dual)** — 8-tool function-calling loop producing L6/L7 and graph edges
- **OpenAI-compatible** — any LLM / embedding endpoint (OpenAI, DashScope, Zhipu, local vLLM, …)
- **Multi-tenant isolation** — `app_id` + `user_id` + optional `agent_id` / `session_id`

## Modes

| | system1 (default) | dual |
|---|---|---|
| LLM per `add` (fast-write) | ~2 (Gate + Extract; +1 summarize if long) | same + async 1~N |
| Layers written | L0 / L1(SHADOW) / L2 / L3 / L4 | + L5 / L6 / L7 |
| Graph (Kuzu) | off | on |
| System2 | no | ReconcileWorker + ReAct Agent + Sweeper |

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
| **SDK basics** | system1 writes, evolution chains, multi-turn `messages` |
| **System2** | dual `digest`, ReAct distillation, cross-domain schema |
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
- [mem0ai/mem0](https://github.com/mem0ai/mem0) — early exploration in layered memory SDKs
