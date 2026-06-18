# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: In-memory BM25+ scoring over a small candidate pool (stats computed on the
pool, not a global corpus) with bilingual tokenization; scores feed RRF ranking.
"""
import math
import re
from collections import Counter

BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercased ascii words and CJK runs."""
    if not text:
        return []
    return [tok.lower() if tok.isascii() else tok for tok in _TOKEN_RE.findall(text)]


def compute_bm25_scores(
    query_terms: list[str],
    candidate_contents: list[str],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    """Compute a BM25+ score per candidate content; zeros if query or pool is empty."""
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
    """Score (id, content) pairs and return them max-normalized and sorted by BM25 desc."""
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
