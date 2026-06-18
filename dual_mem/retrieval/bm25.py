"""BM25-lite：对内存中有限候选池直接跑 BM25+ 公式排序（忠实复现源码）。

IDF / avgdl 的统计基数为"当前候选池"而非全库；分数最终只用于 RRF 的 rank，
对绝对分数不敏感。支持中英文混合：英文 lowercase，汉字段作为整串 token。
"""

import math
import re
from collections import Counter

BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [tok.lower() if tok.isascii() else tok for tok in _TOKEN_RE.findall(text)]


def compute_bm25_scores(
    query_terms: list[str],
    candidate_contents: list[str],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    """对 candidate_contents 每条计算 BM25+ 分数；query 或池为空返回全零。"""
    if not query_terms or not candidate_contents:
        return [0.0] * len(candidate_contents)

    n = len(candidate_contents)
    tokenized = [tokenize(c) for c in candidate_contents]
    doc_lens = [len(toks) for toks in tokenized]
    avgdl = (sum(doc_lens) / n) or 1.0

    q_terms = list({t for t in query_terms if t})
    df = {t: sum(1 for toks in tokenized if t in toks) for t in q_terms}
    tf_per_doc = [Counter(toks) for toks in tokenized]

    scores: list[float] = []
    for i, _ in enumerate(tokenized):
        dl = doc_lens[i] or 1
        tf_map = tf_per_doc[i]
        score = 0.0
        for t in q_terms:
            tf = tf_map.get(t, 0)
            if tf == 0:
                continue
            idf = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1)) / denom
        scores.append(score)

    return scores


def score_and_rank(
    query_terms: list[str],
    candidates: list[tuple[str, str]],
) -> list[tuple[str, float]]:
    """输入 [(id, content), ...]，返回按 BM25 降序、max 归一化的 [(id, score), ...]。"""
    contents = [c for _, c in candidates]
    raw = compute_bm25_scores(query_terms, contents)
    max_s = max(raw) if raw else 0.0
    if max_s <= 0:
        norm = [0.0] * len(raw)
    else:
        norm = [s / max_s for s in raw]
    ranked = [(cid, norm[i]) for i, (cid, _) in enumerate(candidates)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
