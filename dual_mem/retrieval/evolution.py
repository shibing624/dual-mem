# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Evolution-chain expansion for search hits: traces full supersedes chains,
represents each by its head node and de-duplicates same-chain hits keeping the best score.
"""
from dual_mem.types import MemoryNode


def _time_key(node: MemoryNode) -> int:
    """Sort key for a node: memory_at, else gmt_created, else 0."""
    return node.memory_at or node.gmt_created or 0


def _node_to_chain_item(node: MemoryNode) -> dict:
    """Serialize a node into a compact evolution-chain entry."""
    return {
        "node_id": node.node_id,
        "content": node.content,
        "memory_at": node.memory_at,
        "gmt_created": node.gmt_created,
        "speculate": node.speculate,
        "layer": node.layer.value,
    }


def _trace_full_chain(vector, start_node: MemoryNode) -> list[MemoryNode]:
    """Bidirectionally trace the full chain from any node; head at index 0."""
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
    """Expand a single node into its chain representation, or None if it is not chained."""
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
    """Expand evolution chains in a batch of hits and de-duplicate same-chain entries."""
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
