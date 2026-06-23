# 安装

需要 **Python ≥ 3.11**。

## PyPI

```bash
pip install dual-mem              # 核心 SDK（MemoryClient）
pip install dual-mem[all]         # SDK + REST + CLI + MCP
pip install dual-mem[api]         # REST（fastapi + uvicorn）
pip install dual-mem[cli]         # CLI（typer）
pip install dual-mem[mcp]         # MCP server（mcp）
```

## 源码开发

```bash
git clone https://github.com/shibing624/dual-mem.git
cd dual-mem
pip install -e ".[dev]"
```

## 依赖说明

**核心依赖**（默认安装）：`pydantic` / `pydantic-settings` / `pyyaml` / `openai` / `chromadb` / `kuzu` / `scikit-learn` / `numpy`。

**可选 extras**：`api` → fastapi + uvicorn；`cli` → typer；`mcp` → mcp；`all` → 三者兼有。

## 配置

首次 `MemoryClient(...)` 或 `Settings()` 会在 **`~/.dual_mem/config.yaml`** 自动创建默认配置（与包内 `dual_mem/config.default.yaml` 一致），只需填入 API key：

```yaml
llm_api_key: sk-...
embed_api_key: sk-...
```

**dual-mem 同时需要 LLM 与 Embedding 两套 API key**，缺失时 `MemoryClient(...)` 会抛 `MissingCredentialsError`。

配置优先级：显式传参 > `DUAL_MEM_*` 环境变量 > YAML > 默认值。
