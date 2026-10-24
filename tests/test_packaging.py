# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Core modules only import packages declared in [project].dependencies.
"""
import ast
import importlib.metadata as md
import re
import subprocess
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


# Each extra's entry modules must import with only that extra installed. Modules from
# the other extras are blocked so a stray cross-extra import fails here, not on PyPI.
# Entries are executed, not just imported: MCP builders import `mcp` lazily inside.
_EXTRA_ENTRY_POINTS = {
    "mcp": (
        ["fastapi", "uvicorn", "typer"],
        [
            "from dual_mem.mcp.server import build_mcp; build_mcp(ops=object())",
            "import tempfile; from pathlib import Path; "
            "from dual_mem.memhub.mcp import build_hub_mcp; "
            "build_hub_mcp(hub_dir=Path(tempfile.mkdtemp()))",
        ],
    ),
    "api": (["mcp", "typer"], ["from dual_mem.api import create_app; create_app"]),
}


def _run_with_blocked(blocked: list[str], statements: list[str]) -> subprocess.CompletedProcess:
    script = (
        "import sys\n"
        f"blocked = {blocked!r}\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in blocked:\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}')\n"
        "sys.meta_path.insert(0, Block())\n"
        + "".join(f"{stmt}\n" for stmt in statements)
    )
    return subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )


def test_each_extra_imports_without_the_other_extras() -> None:
    for extra, (blocked, statements) in _EXTRA_ENTRY_POINTS.items():
        proc = _run_with_blocked(blocked, statements)
        assert proc.returncode == 0, f"[{extra}] extra alone fails:\n{proc.stderr}"


def test_blocking_a_needed_module_is_detected() -> None:
    # Guards the guard: blocking mcp must break the MCP entry, or the test above proves nothing.
    proc = _run_with_blocked(["mcp"], _EXTRA_ENTRY_POINTS["mcp"][1][:1])
    assert proc.returncode != 0 and "No module named 'mcp" in proc.stderr


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
