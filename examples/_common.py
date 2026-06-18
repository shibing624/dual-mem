"""examples 公共工具：每个 demo 用独立干净的本地存储目录。

配置（API key / base_url / 模型）从 ~/.dual_mem/config.yaml 读取，
这里只覆盖 mode 和 storage_dir。
"""
import shutil
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
DATA_ROOT = EXAMPLES_DIR / ".data"


def fresh_storage(name: str) -> str:
    """返回一个干净的存储目录（每次运行清空，保证 demo 可复现）。"""
    path = DATA_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def show_memories(memories: dict) -> None:
    for group in ("profile", "proactive", "normal"):
        items = memories.get(group) or []
        print(f"  [{group}] {len(items)} 条")
        for m in items:
            chain = m.get("evolution_chain")
            tail = f"   ↳ 演化链(最新→最旧): {chain}" if chain else ""
            print(f"    - ({m.get('category')}) {m.get('content')}{tail}")
