"""Demo 4 — CLI：subprocess 调用 `python -m dual_mem.cli.main`。

临时配置只保留 Settings 认识的键，并改 mode 和 storage_dir。
已删除的旧键不会带进去。JEV 凭据来自配置或 DUAL_MEM_JEV_API_KEY。

环境要求：jev_*、llm_*、embed_*。

运行：python examples/04_cli.py
期望：add / search / search-conversation / list 都返回 0，且 stdout 非空。
"""
import os
import subprocess
import sys
from pathlib import Path

from _common import example_settings, section


def run(args: list[str]) -> str:
    """以子进程方式跑 CLI，返回 stdout。失败时打印 stderr 并退出。"""
    cmd = [sys.executable, "-m", "dual_mem.cli.main", *args]
    out = subprocess.run(cmd, capture_output=True, text=True, env=ENV, cwd=str(Path.cwd()))
    if out.returncode != 0:
        print("STDERR:", out.stderr)
        raise SystemExit(f"命令失败: {' '.join(args)}")
    return out.stdout.strip()


def main() -> None:
    global ENV
    example_settings("system1", "cli")
    ENV = os.environ.copy()
    section("add")
    for text in ["用户喜欢用 vim 编辑器", "用户习惯每天早上 7 点跑步"]:
        print(run(["add", "--content", text, "--user-id", "erin"]))

    section("search（当前 L1/L2，不含 L0 原文）")
    print(run(["search", "用户的运动习惯是什么？", "--user-id", "erin", "--json"]))

    section("search-conversation（只查 L0_RAW）")
    print(run(["search-conversation", "跑步", "--user-id", "erin", "--json"]))

    section("list（默认 latest）")
    print(run(["list", "--user-id", "erin"]))


if __name__ == "__main__":
    main()
