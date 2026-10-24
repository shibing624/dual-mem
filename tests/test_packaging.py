# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Core modules only import packages declared in [project].dependencies.
"""
import ast
import importlib.metadata as md
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Frontends shipped behind optional extras; everything else must run on a bare install.
OPTIONAL_SUBPACKAGES = {"api", "cli", "mcp", "memhub", "integrations"}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _core_imports() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in (ROOT / "dual_mem").rglob("*.py"):
        rel = path.relative_to(ROOT / "dual_mem")
        if rel.parts[0] in OPTIONAL_SUBPACKAGES:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top != "dual_mem":
                    found.setdefault(top, set()).add(str(rel))
    return found


def test_core_third_party_imports_are_declared_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {
        _normalize(re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0])
        for spec in project["dependencies"]
    }
    distributions = md.packages_distributions()

    missing = {}
    for module, files in _core_imports().items():
        dists = {_normalize(d) for d in distributions.get(module, [module])}
        if not dists & declared:
            missing[module] = sorted(files)
    assert not missing, f"undeclared core dependencies: {missing}"
