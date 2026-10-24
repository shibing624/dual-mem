<h1 align="center">dual-mem</h1>

<p align="center"><b>Facts now. Patterns over time.</b><br />System 1 writes facts and changes while you talk. dual (System 2) turns them into stable patterns. MemHub lets several agents share that one memory.</p>

<p align="center">
  <a href="https://badge.fury.io/py/dual-mem"><img src="https://badge.fury.io/py/dual-mem.svg" alt="PyPI version" /></a>
  <a href="https://github.com/shibing624/dual-mem/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License Apache 2.0" /></a>
  <a href="https://github.com/shibing624/dual-mem/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-green.svg" alt="Python 3.11+" /></a>
  <a href="https://github.com/shibing624/dual-mem/issues"><img src="https://img.shields.io/github/issues/shibing624/dual-mem.svg" alt="GitHub issues" /></a>
  <a href="#community--support"><img src="https://img.shields.io/badge/wechat-group-green.svg?logo=wechat" alt="Wechat Group" /></a>
</p>

<p align="center"><a href="https://github.com/shibing624/dual-mem/blob/main/README.md">简体中文</a> | English | <a href="https://shibing624.github.io/dual-mem">Documentation</a></p>

## Why dual-mem

You've told your assistant three times that you no longer eat spicy food, and it still recommends a Sichuan restaurant. After dozens of turns it can quote what you said, and still cannot name the habit that shows up every time. The coding conventions you explained in Cursor are unknown to Codex.

### 1. Same model, sharper memory, faster writes

One model (Qwen3.8-27B) does extraction, answering and grading. One embedding model (bge-m3). The same questions. **Only the memory system changes**:

| Memory system | LoCoMo<br>(308 q) | LongMemEval<br>(24 q) | PersonaMem-v2<br>(476 q) | Ingestion time per long dialogue<br>(s, lower is better) |
|---|:---:|:---:|:---:|:---:|
| **dual-mem** | **74.7** | **91.7** | **50.0** | **503** |
| hy-memory 2.0.4 | 66.6 | 70.8 | 43.5 | 2485 |
| mem0 | 65.6 | 83.3 | 47.9 | 808 |

**First on all three public benchmarks, and about one fifth the ingestion time of hy-memory.** Protocol, significance tests, per-category results and reproduction commands: [Benchmarks](https://github.com/shibing624/dual-mem/blob/main/docs/benchmarks.md).

### 2. Two systems. That is the dual.

The name means two chat-memory systems. That is the method a paper should describe.

- **System 1 (default `system1`)** writes while the conversation is happening. Raw words, facts, and profile live in separate layers. When you change your mind, move cities, or update a plan, it appends a new version and keeps the history. Answers use the latest.
- **dual / System 2 (`mode="dual"`, then `digest()`)** runs after the session. It induces behavioral patterns and a longer-term profile (L3) across many pieces of evidence.

See [`examples/02_dual.py`](https://github.com/shibing624/dual-mem/blob/main/examples/02_dual.py).

### 3. Several agents, one memory

MemHub connects those two systems to the tools on your machine. Claude Code, Codex, Cursor, Hermes, and Agentica read and write the same memory. By default the data stays on your computer.

```bash
pip install "dual-mem[all]"

dual-mem hub install-service       # runs in the background, starts at login
dual-mem hub install-host all      # connect every supported AI tool on this machine
dual-mem hub scan --source all     # turn your past conversations into memory

dual-mem hub recall "what technical choices have I made before"
```

Tools discover the local service with no URL to paste. A conversation is saved and acknowledged immediately; organizing happens in the background. When a turn ends, the host's hook hands the new messages to the Core, and the next prompt you type gets the relevant memories injected; agents can also query through MCP. Coding sessions arrive with their tool calls and become reusable engineering experience.

To share one memory across computers, run the only Core on your server and connect each computer with one command:

```bash
dual-mem hub connect user@your-server   # builds the SSH tunnel, reads the server token, writes local hub.json
```

The Core still listens only on the server's `127.0.0.1`, so no port is exposed; the tunnel runs as a login service and reconnects on its own. There is always exactly one store — no push / pull merging. See [server setup](https://github.com/shibing624/dual-mem/blob/main/docs/memhub-server.md) and the [MemHub docs](https://github.com/shibing624/dual-mem/blob/main/docs/memhub.md).

### 4. Debugging sessions stay useful

Chat and coding share one `add()` / `search()`. When `messages` carry a tool trace (a `role="tool"` turn or assistant `tool_calls`) and JEV calls the session engineering work, the write also yields reusable engineering experience (`kind="procedure"`: task, solution, files, boundary) next to the usual facts and profile. Procedures share the store, the version chain and JEV adjudication with chat memories; `search(procedure_limit=3)` returns them as a separate `procedures` group, and in `dual` mode they feed `digest()` too. Details: [Quickstart · Coding memory](https://github.com/shibing624/dual-mem/blob/main/docs/getting-started/quickstart.md#coding).

## Install

```bash
pip install dual-mem
```

REST / CLI / MCP / MemHub: `pip install "dual-mem[all]"`; each extra also installs on its own, and MCP runs on mcp 2.x. The first run copies the packaged [`config.default.yaml`](https://github.com/shibing624/dual-mem/blob/main/dual_mem/config.default.yaml) to `~/.dual_mem/config.yaml`. Construction needs three keys; a missing one refuses to start:

```yaml
jev_api_key: ""    # write decisions. System One, not Chat Completions
jev_endpoint: https://api.typesafe.ai/v1/systemone

llm_api_key: ""    # extraction and System 2. OpenAI-compatible; base URL is yours
llm_base_url: https://api.openai.com/v1

embed_api_key: ""  # vector index and search. OpenAI-compatible; base URL is yours
embed_base_url: https://api.openai.com/v1
```

The same six values are environment variables: `DUAL_MEM_JEV_API_KEY` / `DUAL_MEM_JEV_ENDPOINT`, `DUAL_MEM_LLM_API_KEY` / `DUAL_MEM_LLM_BASE_URL`, `DUAL_MEM_EMBED_API_KEY` / `DUAL_MEM_EMBED_BASE_URL`.

## Quick start

### Everyday memory

```python
from dual_mem import SyncMemoryClient

with SyncMemoryClient() as memory:
    memory.add(content="I live in Shanghai and recently stopped eating spicy food.", user_id="alice")
    memory.add(content="I'm moving to Beijing for work next month.", user_id="alice")

    res = memory.search(query="recommend a restaurant for alice", user_id="alice")
    for m in res.memories.profile + res.memories.normal:
        print(m.content)
```

It remembers that alice no longer eats spicy food and is about to move to Beijing, and it knows Shanghai is becoming the past.

### Coding memory

No switch and no second search entry point. A tool trace in `messages` (a `role="tool"` turn, or assistant `tool_calls`) is the structural condition; JEV then decides whether the session is engineering work, and only then extracts `task` / `solution` / `files` as an L1 memory with `kind="procedure"`. Preferences the user mentions in the same session ("I always use uv") still become facts and profile.

Putting the text of a few `.py` files into `add(content=...)` or a plain user message has no tool trace, so it stays chat memory only.

```python
from dual_mem import SyncMemoryClient

with SyncMemoryClient() as memory:
    memory.add(
        messages=[
            {"role": "user", "content": "Fix intermittent sqlite3.OperationalError: database is locked under concurrent writes."},
            {
                "role": "assistant",
                "content": "Let me inspect how transactions are opened.",
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
        workspace_id="~/code/app",  # version boundary of the procedure; optional
    )

    res = memory.search(
        query="how to fix SQLite database is locked on concurrent writes",
        user_id="alice",
        procedure_limit=3,  # default 0: chat results are exactly as before
    )
    for m in res.memories.procedures:
        print(m.procedure.task, "->", m.procedure.solution, m.procedure.files)
```

### Examples

Two worth opening: [`examples/02_dual.py`](https://github.com/shibing624/dual-mem/blob/main/examples/02_dual.py) runs `digest()` and builds L3, and [`examples/06_coding_memory.py`](https://github.com/shibing624/dual-mem/blob/main/examples/06_coding_memory.py) writes a tool trace, evolves a solution, and reads the `procedures` group. REST, CLI, and lifecycle scripts are in [`examples/`](https://github.com/shibing624/dual-mem/tree/main/examples). Async clients: [Quickstart](https://github.com/shibing624/dual-mem/blob/main/docs/getting-started/quickstart.md).

## Features

**Memory**

- **Versioned updates** — "lives in Shanghai → moving to Beijing" is an update. The old version stays as history. Answers use the latest.
- **Separate layers** — raw words (L0), facts and intentions (L1), profile (L2), cross-evidence patterns (L3), retrieved as needed.
- **A decision model guards writes** — new, duplicate, or update is judged by a lightweight model.
- **The original words stay available** — raw conversation is kept alongside the organized memory.

**Ways in**

- **Python SDK** — `pip install dual-mem`
- **MemHub** — `dual-mem hub install-host all`
- **MCP / REST / CLI** — `dual-mem-mcp`, `dual-mem serve`, `dual-mem add` / `dual-mem search`
- **Framework plugins** — Agentica, Claude Code hooks, Hermes, OpenClaw. See [Integrations](https://github.com/shibing624/dual-mem/blob/main/docs/integrations.md).

## Roadmap

- [x] System 1 layered memory and version evolution
- [x] dual / System 2: explicit `digest()` builds L3
- [x] Dedicated decision model guarding writes
- [x] MemHub: 5 AI tools share memory, history import, background organizing
- [x] Single-instance server mode: several computers `hub connect` to one memory
- [x] Chat and coding unified: coding sessions yield procedure experience, and `dual` induces engineering habits too
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

- [2026/10/24] **v0.2.0**: Chat and coding now share one `add()` / `search()`. An engineering session with a tool trace yields `kind="procedure"` experience next to its chat memories, on the same store, version chain and JEV adjudication; `search(procedure_limit=3)` returns them as a separate `procedures` group, and `dual` `digest()` covers them too. `search_coding` and the separate coding store are gone. Chat prompts, adjudication and recall are unchanged. MemHub captures tool traces from Claude Code / Codex / Agentica and auto-digests when idle in dual mode; new `dual-mem hub connect user@server` sets up the tunnel, reads the token and writes `hub.json` in one command. MCP moves to mcp 2.x; `install-host` keeps YAML comments. See [Release-v0.2.0](https://github.com/shibing624/dual-mem/releases/tag/v0.2.0).
- [2026/10/23] **v0.1.5**: Fixed Coding memory writes against real models. Extraction now returns a `{"candidates": [...]}` object so the default `llm_json_mode` keeps the payload; the coverage hard gate is gone; one batch can supersede the same head; JEV only sees heads inside the draft's version boundary; concurrent writes for one user serialize per scope. The quickstart now has a Coding memory section, and `examples/06_coding_memory.py` shows search returning only the current solution after an update. See [Release-v0.1.5](https://github.com/shibing624/dual-mem/releases/tag/v0.1.5).
- [2026/10/23] **v0.1.4**: Added MemHub local single-Core cross-tool memory hub (incremental session import for Claude Code / Codex / Cursor / Hermes / Agentica, durable background write queue, login service, and SSH/Tailscale tunnel attach); upgraded System 2 v3 clustered cognition and version-bridge recall; added Coding memory runnable demo and removed zero-gain settings; ranked #1 head-to-head on LoCoMo (74.7), LongMemEval (91.7), and PersonaMem-v2 (50.0). See [Release-v0.1.4](https://github.com/shibing624/dual-mem/releases/tag/v0.1.4).
- [2026/10/15] **v0.1.3**: Redesigned around four canonical layers (`L0_RAW` / `L1_FACT` / `L2_PROFILE` / `L3_DERIVED`) and target-first JEV write decisions with linked version history; added native integrations for Agentica / Claude Code / Hermes / OpenClaw and the Coding memory subsystem (incompatible storage schema; use a new `storage_dir`). See [Release-v0.1.3](https://github.com/shibing624/dual-mem/releases/tag/v0.1.3).

<details>
<summary>Earlier releases</summary>

- [2026/06/20] **v0.1.2**: Added `SyncMemoryClient` synchronous wrapper, automatic `~/.dual_mem/config.yaml` bootstrap on first run, unified REST / MCP / CLI contracts, and the online documentation site. See [Release-v0.1.2](https://github.com/shibing624/dual-mem/releases/tag/v0.1.2).
- [2026/06/19] **v0.1.1**: Released dual-system layered memory SDK with hybrid retrieval (semantic + BM25 + RRF) and CLI / REST / MCP interfaces. See [Release-v0.1.1](https://github.com/shibing624/dual-mem/releases/tag/v0.1.1).
- [2026/06/18] **v0.1.0**: First open-source release.

</details>

Full history in the [CHANGELOG](https://github.com/shibing624/dual-mem/blob/main/CHANGELOG.md).

## Docs

Full documentation: **[https://shibing624.github.io/dual-mem](https://shibing624.github.io/dual-mem)**

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

Local preview: `pip install -r requirements-docs.txt && mkdocs serve`

## Community & support

> If dual-mem helps, a ⭐ star helps more people find it.

- **GitHub Issues** — [open an issue](https://github.com/shibing624/dual-mem/issues)
- **WeChat group** — add `xuming624`, note "llm"

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" alt="WeChat group QR" />

## Citation

If you use dual-mem in research, please cite:

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

## License

[Apache License 2.0](https://github.com/shibing624/dual-mem/blob/main/LICENSE)

## Contributing

Contributions welcome. See [CONTRIBUTING.md](https://github.com/shibing624/dual-mem/blob/main/CONTRIBUTING.md).

## Acknowledgements

- [Chroma](https://github.com/chroma-core/chroma)
- [Kuzu](https://github.com/kuzudb/kuzu)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [mem0](https://github.com/mem0ai/mem0)
