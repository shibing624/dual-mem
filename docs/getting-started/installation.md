# 安装

需要 **Python ≥ 3.11**。

## PyPI

```bash
pip install dual-mem              # 核心 SDK（MemoryClient）
pip install dual-mem[all]         # SDK + REST + CLI + MCP
pip install dual-mem[api]         # REST（fastapi + uvicorn）
pip install dual-mem[cli]         # CLI（typer）
pip install dual-mem[mcp]         # MCP server（mcp 2.x，可单独安装）
```

## 源码开发

```bash
git clone https://github.com/shibing624/dual-mem.git
cd dual-mem
pip install -e ".[dev]"
```

## 依赖说明

**核心依赖**（默认安装）：`pydantic` / `pydantic-settings` / `pyyaml` / `ruamel.yaml` / `openai` / `chromadb` / `kuzu` / `scikit-learn` / `numpy` / `httpx`。`scikit-learn` 供 `dual` 模式的 `digest()` 聚类使用；`ruamel.yaml` 供 `dual-mem hub install-host` 改写宿主 YAML 时保留注释。

**可选 extras**：`api` → fastapi + uvicorn；`cli` → typer；`mcp` → mcp 2.x（`mcp>=2.2,<3`）；`all` → 三者兼有。每个 extra 可以单独安装，例如只装 `[mcp]` 也能运行 `dual-mem-mcp`。

## 配置

首次 `MemoryClient(...)` 或 `Settings()` 会在 **`~/.dual-mem/config.yaml`** 自动创建默认配置（与包内 `dual_mem/config.default.yaml` 一致），只需填入 API key：

```yaml
jev_api_key: ...
llm_api_key: sk-...
embed_api_key: sk-...
```

构造需要三组凭证：JEV（`jev_api_key`）、生成 LLM（`llm_api_key`）、Embedding（`embed_api_key`）。缺任何一组都会抛 `MissingCredentialsError`。测试或自建后端可以注入 `jev=` / `llm=` / `embed=`，注入后跳过对应密钥检查。

配置优先级：显式传参 > `DUAL_MEM_*` 环境变量 > YAML > 默认值。只接受平铺键；已删除的旧键会让 `Settings` 直接报错。默认 `storage_dir` 是 `./.dual_mem_v4`。旧的 `.dual_mem_data` 不会被打开，也不会被迁移；配置里如果还写着旧路径，打开时仍会因 schema 不匹配报错且不改写文件。
