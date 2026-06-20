"""Demo 4 — CLI：通过 `dual-mem` 命令行真实调用。

用 subprocess 调用已安装的 CLI（system1 档），演示 add / search / list 子命令。
配置（API key 等）从 `~/.dual_mem/config.yaml` 读取；本 demo 派生一份临时配置，
只覆盖 `mode=system1` 与 `storage_dir`，避免污染默认数据目录。

环境要求：`llm_*` + `embed_*` 都齐全。

运行：python examples/04_cli.py
期望：每条命令返回非空 JSON / 结构化文本输出。
"""
import os
import subprocess
import sys
from pathlib import Path

import yaml
from _common import EXAMPLES_DIR, fresh_storage, section

from dual_mem.config import config_path

# 基于真实 yaml 派生一份临时配置，只改 mode + storage_dir，保证 demo 干净可复现。
base_cfg = yaml.safe_load(config_path().read_text(encoding="utf-8")) or {}
base_cfg["mode"] = "system1"
base_cfg["storage_dir"] = fresh_storage("cli")
tmp_cfg = EXAMPLES_DIR / ".data" / "cli_config.yaml"
tmp_cfg.write_text(yaml.safe_dump(base_cfg, allow_unicode=True), encoding="utf-8")

ENV = {**os.environ, "DUAL_MEM_CONFIG_FILE": str(tmp_cfg)}


def run(args: list[str]) -> str:
    """以子进程方式跑 dual-mem CLI，返回 stdout（失败时抛错）。"""
    cmd = [sys.executable, "-m", "dual_mem.cli.main", *args]
    out = subprocess.run(cmd, capture_output=True, text=True, env=ENV, cwd=str(Path.cwd()))
    if out.returncode != 0:
        print("STDERR:", out.stderr)
        raise SystemExit(f"命令失败: {' '.join(args)}")
    return out.stdout.strip()


def main() -> None:
    section("dual-mem add（写入两条，省略 --app-id）")
    for text in ["用户喜欢用 vim 编辑器", "用户习惯每天早上 7 点跑步"]:
        print(run(["add", "--content", text, "--user-id", "erin"]))

    section("dual-mem search（检索作息习惯）")
    print(run(["search", "用户的运动习惯是什么？", "--user-id", "erin"]))

    section("dual-mem list")
    print(run(["list", "--user-id", "erin"]))


if __name__ == "__main__":
    main()
