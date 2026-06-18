"""两阶段 DBSCAN 事实聚类（System2 预处理的硬编码步骤，无 LLM）。

算法忠实复现 hy-memory ``prepare_materials`` 的聚类逻辑：
- Stage1：粗分，eps = 1 - 0.55，min_samples = 3。
- Stage2：对 size > 12 的大簇细分，eps = 1 - 0.75，min_samples = 2，
  noise 归入最近的子簇。
- 簇内 cosine >= 0.92 去重，保留离质心最近的代表。

用 cosine 预计算距离矩阵（distance = 1 - cosine）。
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from dual_mem.types import MemoryNode

_MIN_FACTS = 3
_STAGE2_MIN_SAMPLES = 2
_MAX_CLUSTER_SIZE = 12
_DEDUP_COSINE = 0.92


def cluster_facts(
    facts: list[MemoryNode],
    *,
    stage1_sim: float = 0.55,
    stage2_sim: float = 0.75,
) -> list[dict]:
    pool = [f for f in facts if f.embedding]
    if len(pool) < _MIN_FACTS:
        return []

    stage1_eps = 1.0 - stage1_sim
    stage2_eps = 1.0 - stage2_sim

    matrix = np.array([f.embedding for f in pool])
    labels = DBSCAN(
        eps=stage1_eps, min_samples=_MIN_FACTS, metric="precomputed"
    ).fit_predict(cosine_distances(matrix))

    stage1: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        stage1.setdefault(label, []).append(idx)

    groups: list[list[int]] = []
    for indices in stage1.values():
        if len(indices) <= _MAX_CLUSTER_SIZE:
            groups.append(indices)
        else:
            groups.extend(_refine(matrix, indices, stage2_eps))

    clusters: list[dict] = []
    for indices in groups:
        if len(indices) < _STAGE2_MIN_SAMPLES:
            continue
        kept = _dedup(matrix, indices)
        centroid = np.mean(matrix[kept], axis=0)
        rep = min(kept, key=lambda i: float(np.linalg.norm(matrix[i] - centroid)))
        clusters.append(
            {
                "ids": [pool[i].node_id for i in kept],
                "centroid_text": pool[rep].content,
                "centroid_embedding": centroid.tolist(),
                "facts": [
                    {
                        "node_id": pool[i].node_id,
                        "content": pool[i].content,
                        "layer": pool[i].layer.value,
                    }
                    for i in kept
                ],
            }
        )
    return clusters


def _refine(matrix: np.ndarray, indices: list[int], stage2_eps: float) -> list[list[int]]:
    sub = matrix[indices]
    sub_labels = DBSCAN(
        eps=stage2_eps, min_samples=_STAGE2_MIN_SAMPLES, metric="precomputed"
    ).fit_predict(cosine_distances(sub))

    sub_groups: dict[int, list[int]] = {}
    noise: list[int] = []
    for pos, label in enumerate(sub_labels):
        if label == -1:
            noise.append(indices[pos])
        else:
            sub_groups.setdefault(label, []).append(indices[pos])

    keys = list(sub_groups.keys())
    result = [sub_groups[k] for k in keys]
    if noise and result:
        centroids = {k: np.mean(matrix[sub_groups[k]], axis=0) for k in keys}
        for ni in noise:
            nearest = min(
                keys,
                key=lambda k: float(
                    cosine_distances(matrix[ni : ni + 1], centroids[k].reshape(1, -1))[0, 0]
                ),
            )
            result[keys.index(nearest)].append(ni)
    elif noise:
        result.append(noise)
    return result


def _dedup(matrix: np.ndarray, indices: list[int]) -> list[int]:
    kept: list[int] = []
    for i in indices:
        duplicate = any(
            1.0 - float(cosine_distances(matrix[i : i + 1], matrix[k : k + 1])[0, 0])
            >= _DEDUP_COSINE
            for k in kept
        )
        if not duplicate:
            kept.append(i)
    return kept
