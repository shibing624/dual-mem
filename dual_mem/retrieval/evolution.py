"""演化链回溯（忠实复现 hy_memory _retrieval/evolution，适配同步 API）。

对一批 search hits，识别其中在演化链上的节点，双向追溯完整链，以链头
（is_latest=True；否则 gmt_created/memory_at 最新）为代表返回，同链多个 hit
去重合并（保留最高 score）。
"""

from dual_mem.types import MemoryNode


def _time_key(node: MemoryNode) -> int:
    return node.memory_at or node.gmt_created or 0


def _node_to_chain_item(node: MemoryNode) -> dict:
    return {
        "node_id": node.node_id,
        "content": node.content,
        "memory_at": node.memory_at,
        "gmt_created": node.gmt_created,
        "speculate": node.speculate,
        "layer": node.layer.value,
    }


def _trace_full_chain(vector, start_node: MemoryNode) -> list[MemoryNode]:
    """从任意节点出发双向追溯整条链（向前 supersedes、向后 superseded_by）。

    返回整条链，链头在 [0]。只有自身时返回 [start_node]。
    """
    visited: dict[str, MemoryNode] = {start_node.node_id: start_node}
    to_fetch: list[str] = list(start_node.supersedes) + list(start_node.superseded_by)

    while to_fetch:
        ids_batch = [i for i in to_fetch if i not in visited]
        if not ids_batch:
            break
        new_to_fetch: list[str] = []
        for nid in ids_batch:
            node = vector.get(nid)
            if node is None or node.node_id in visited:
                continue
            visited[node.node_id] = node
            new_to_fetch.extend(node.supersedes)
            new_to_fetch.extend(node.superseded_by)
        to_fetch = new_to_fetch

    if len(visited) == 1:
        return [start_node]

    all_nodes = list(visited.values())
    heads = [n for n in all_nodes if n.is_latest]
    head = max(heads, key=_time_key) if heads else max(all_nodes, key=_time_key)
    rest = sorted(
        (n for n in all_nodes if n.node_id != head.node_id),
        key=_time_key,
        reverse=True,
    )
    return [head] + rest


def _expand_one_chain(vector, node: MemoryNode) -> dict | None:
    if not node.supersedes and not node.superseded_by:
        return None
    chain = _trace_full_chain(vector, node)
    if len(chain) <= 1:
        return None
    return {
        "head": chain[0],
        "evolution_chain": [_node_to_chain_item(n) for n in chain],
        "chain_node_ids": {n.node_id for n in chain},
        "score": node.score,
    }


def expand_evolution_chains(*, vector, hits: list[MemoryNode]) -> list[dict]:
    """展开一批 hits 中的演化链并去重。

    输入 hits 已按 node_id 去重。输出每项形如
    ``{"node": MemoryNode, "score": float, "is_evolved": bool,
    "evolution_chain": list|None}``，可能比输入短（同链合并）。
    """
    expanded_by_idx: dict[int, dict] = {}
    chain_dedup: dict[str, dict] = {}
    for i, node in enumerate(hits):
        exp = _expand_one_chain(vector, node)
        if exp is None:
            continue
        head_id = exp["head"].node_id
        existing = chain_dedup.get(head_id)
        if existing is None or exp["score"] > existing["score"]:
            chain_dedup[head_id] = exp
        expanded_by_idx[i] = exp

    all_chain_node_ids: set[str] = set()
    for exp in chain_dedup.values():
        all_chain_node_ids.update(exp["chain_node_ids"])

    result: list[dict] = []
    seen_chain_heads: set[str] = set()
    seen_node_ids: set[str] = set()
    for i, node in enumerate(hits):
        if i in expanded_by_idx:
            head_id = expanded_by_idx[i]["head"].node_id
            if head_id in seen_chain_heads:
                continue
            best = chain_dedup[head_id]
            seen_chain_heads.add(head_id)
            seen_node_ids.add(head_id)
            result.append(
                {
                    "node": best["head"],
                    "score": best["score"],
                    "is_evolved": True,
                    "evolution_chain": best["evolution_chain"],
                }
            )
        else:
            if node.node_id in all_chain_node_ids or node.node_id in seen_node_ids:
                continue
            seen_node_ids.add(node.node_id)
            result.append(
                {
                    "node": node,
                    "score": node.score,
                    "is_evolved": False,
                    "evolution_chain": None,
                }
            )
    return result
