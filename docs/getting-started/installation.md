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
jev_api_key: ...
llm_api_key: sk-...
embed_api_key: sk-...
```

构造需要三组凭证：JEV（`jev_api_key`）、生成 LLM（`llm_api_key`）、Embedding（`embed_api_key`）。缺任何一组都会抛 `MissingCredentialsError`。测试或自建后端可以注入 `jev=` / `llm=` / `embed=`，注入后跳过对应密钥检查。

配置优先级：显式传参 > `DUAL_MEM_*` 环境变量 > YAML > 默认值。只接受平铺键；已删除的旧键会让 `Settings` 直接报错。已有数据目录的 `layer_schema_version` 必须是 4，否则打开即失败且不改写文件，换一个新的 `storage_dir`。
