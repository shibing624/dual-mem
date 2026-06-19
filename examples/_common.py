"""examples 公共工具：每个 demo 用独立干净的本地存储目录。

配置（API key / base_url / 模型）从 ~/.dual_mem/config.yaml 读取，
这里只覆盖 mode 和 storage_dir。
"""
import shutil
from pathlib import Path

from dual_mem.sdk_models import MemoryItem, SearchMemories

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


def _render_chain(item: MemoryItem) -> str:
    """渲染演化链（最新→最旧）的简短文本，未发生演化时返回空串。"""
    chain = item.evolution_chain or []
    if not chain:
        return ""
    parts = [f"{node.layer}:{node.content[:20]}" for node in chain]
    return f"   ↳ 演化链(最新→最旧): {parts}"


def show_memories(memories: SearchMemories) -> None:
    """统一渲染三路召回：profile / proactive / normal。"""
    for group, items in (
        ("profile", memories.profile),
        ("proactive", memories.proactive),
        ("normal", memories.normal),
    ):
        print(f"  [{group}] {len(items)} 条")
        for item in items:
            chain_tail = _render_chain(item)
            print(f"    - ({item.category} score={item.score:.2f}) {item.content}{chain_tail}")
