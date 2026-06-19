# 贡献指南

感谢关注 dual-mem！

## 开发环境

```bash
git clone https://github.com/shibing624/dual-mem.git
cd dual-mem
pip install -e ".[dev]"
```

## 测试

测试全部 mock，不连接真实 LLM / Embed：

```bash
python -m pytest tests/ -q
python ~/.agents/rules/check_ast.py .
```

## 提交 PR

1. Fork 仓库并创建特性分支
2. 保持改动聚焦；匹配现有代码风格（4 空格、类型注解、绝对 import）
3. 确保 `pytest` 与 `check_ast.py` 通过
4. 向 `main` 发起 Pull Request，说明动机与测试方式

## 文档

- 用户文档站点：https://shibing624.github.io/dual-mem
- 修改 `docs/` 或 `mkdocs.yml` 后，合并到 `main` 会自动部署 GitHub Pages

## 问题反馈

请通过 [GitHub Issues](https://github.com/shibing624/dual-mem/issues) 提交 bug 或功能建议。
