[**🇨🇳中文**](https://github.com/shibing624/dual-mem/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/dual-mem/blob/main/README_EN.md)

<div align="center">

# dual-mem

**All your agents. One Memory.**

AI that actually remembers you — it notices when your preferences change, and it still knows you when you switch tools.

[![PyPI version](https://badge.fury.io/py/dual-mem.svg)](https://badge.fury.io/py/dual-mem)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.11%2B-green.svg)](pyproject.toml)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/dual-mem.svg)](https://github.com/shibing624/dual-mem/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#community--support)

</div>

---

You've told your assistant three times that you no longer eat spicy food, and it still recommends a Sichuan restaurant.
The coding conventions you explained in Cursor are unknown to Codex; the stack you settled on in Claude Code has to be re-explained in Hermes.

**dual-mem fixes both:**

- 🧠 **Remembers accurately** — instead of piling up chat logs, it organizes memory the way people do: facts, profile, and habits are kept apart, and when you change your mind it updates rather than getting confused.
- 🔗 **Remembers everywhere** — MemHub lets the AI tools you use share one personal memory. Your memory is a long-term asset that no single tool owns — every conversation starts from knowing you.

## Results: head-to-head, same setup

The same model (Qwen3.8-27B) does extraction, answering and grading; same embeddings, same questions — **only the memory system changes**:

| Memory system | LoCoMo<br>(308 q) | LongMemEval<br>(24 q) | PersonaMem-v2<br>(476 q) | Write time<br>(s / conversation) |
|---|:---:|:---:|:---:|:---:|
| **dual-mem** | **74.7** | **91.7** | **50.0** | **503** |
| hy-memory 2.0.4 | 66.6 | 70.8 | 43.5 | 2485 |
| mem0 | 65.6 | 83.3 | 47.9 | 808 |

> Write time is the median per LoCoMo long conversation. Protocol, significance tests, per-category results and reproduction commands: [Benchmarks](docs/benchmarks.md).

## Get started in 3 minutes

```bash
pip install dual-mem
```

The first run creates `~/.dual_mem/config.yaml`; fill in API keys for your model services (any OpenAI-compatible endpoint).

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

Building your own agent? See the [Quickstart](docs/getting-started/quickstart.md) for the async client, multi-turn writes and agent integration.

## MemHub: one memory for all your AI tools

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

Supported today: **Claude Code, Codex, Cursor, Hermes, Agentica**. More tools are on the [Roadmap](#roadmap). See the [MemHub docs](docs/memhub.md).

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

Curious how it works? See the [developer docs](#developer-docs).

## Ways to integrate

| Option | Best for | One-liner |
|---|---|---|
| **Python SDK** | your own agent or app | `pip install dual-mem` |
| **MemHub** | daily use across several AI tools | `dual-mem hub install-host all` |
| **MCP** | Cursor, Claude Desktop and other MCP clients | `dual-mem-mcp` |
| **REST API** | other languages or a standalone service | `dual-mem serve` |
| **CLI** | scripts and debugging | `dual-mem add` / `dual-mem search` |
| **Framework plugins** | Agentica, Claude Code hooks, Hermes, OpenClaw | see [Integrations](docs/integrations.md) |

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

## 🔥 News

- [2026/10/23] LoCoMo head-to-head: dual-mem 74.7, more than 8 points ahead of hy-memory and mem0.
- [2026/10/22] Leaner configuration: switches with no measured benefit removed; defaults are the best settings.
- [2026/10/15] **v0.1.3**: new layered memory with decision-model writes (storage format incompatible with older versions; use a new directory).
- [2026/06/18] **v0.1.0**: first open-source release.

Full history in the [CHANGELOG](CHANGELOG.md).

## Developer docs

<details>
<summary>Expand: architecture, configuration and internals (optional)</summary>

| Doc | Contents |
|---|---|
| [Installation & configuration](docs/getting-started/installation.md) | extras, config file, credentials and precedence |
| [Quickstart](docs/getting-started/quickstart.md) | sync / async clients, multi-turn writes, agent integration |
| [Architecture](docs/architecture.md) | how the SDK relates to each frontend; storage |
| [Memory layers](docs/memory_layers.md) | L0 raw / L1 facts / L2 profile / L3 derived |
| [Write decisions](docs/jev-memory-design.md) | decision-model actions and relations, version chains, concurrency |
| [MemHub](docs/memhub.md) | Core service, host integration, server mode |
| [MCP / REST](docs/mcp_integration.md) | tool ↔ HTTP ↔ SDK mapping |
| [Integrations](docs/integrations.md) | Agentica / Claude Code / Hermes / OpenClaw |
| [Benchmarks](docs/benchmarks.md) | protocol, question sets, reproduction, breakdowns |
| [Examples](examples/README.md) | runnable end-to-end examples |

Online docs: **https://shibing624.github.io/dual-mem** · Local preview: `pip install -r requirements-docs.txt && mkdocs serve`

</details>

## Community & Support

- **GitHub Issues** — [open an issue](https://github.com/shibing624/dual-mem/issues)
- **WeChat group** — add WeChat ID `xuming624` with the note "llm" to join

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" />

## Citation

If you use dual-mem in your research, please cite:

> Xu, M. (2026). dual-mem: Dual-System Layered Memory SDK for LLM Applications. GitHub. https://github.com/shibing624/dual-mem

## License · Contributing · Acknowledgements

[Apache License 2.0](LICENSE) · Contributions welcome, see [CONTRIBUTING.md](CONTRIBUTING.md)

Thanks to [Chroma](https://github.com/chroma-core/chroma), [Kuzu](https://github.com/kuzudb/kuzu), [FastAPI](https://github.com/tiangolo/fastapi), [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) and [mem0](https://github.com/mem0ai/mem0).
